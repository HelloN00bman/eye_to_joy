import argparse
import datetime
import numpy as np
import os
import os.path
import pickle
import progress.bar as bar
import socket
import time
import torch
import torch.nn as nn
import torch.optim as optim

from eye_to_joy import utils
from eye_to_joy import GazeEncoderMS
from eye_to_joy import GazeDataset
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

def mask(pred, label, mini=0, maxi=1):
	mask = (((label[:,0]<=maxi) * (label[:,0]>=mini)) + ((label[:,1]<=maxi) * (label[:,1]>=mini))).float()
	mask[mask>0] = mask[mask>0] - 1
	mask = mask.unsqueeze(len(label.size())-1).expand_as(label).float()
	num = torch.sum(mask) / 2
	print(mask.shape)
	print(num)
	num[num == 0] = 1
	return pred*mask, label*mask, num

def message(msg):
	return '[' + datetime.datetime.now().strftime('%b-%d-%y %H:%M:%S') + ']\t' + msg

def copy_loss(gaze3, label):
	### The loss if you just copy the last element of the gaze vector. ###
	pred = gaze3[:, :, -1]
	pred, _, _ = mask(pred, pred)
	pred_mask, label_mask, num = mask(pred, label)
	loss = torch.nn.MSELoss(size_average = False)(pred_mask, label_mask) / num
	return loss

def const_vel_loss(gaze3, label, mini=0, maxi=1):
	### Calculate the velocity over gaze3. ###
	m = (((gaze3[:,0,:]<=maxi) * (gaze3[:,0,:]>=mini)) + ((gaze3[:,1,:]<=maxi) * (gaze3[:,1,:]>=mini))).float()
	m[m>0] = m[m>0] - 1
	m = m.unsqueeze(len(gaze3.size())-2).expand_as(gaze3).float()
	gaze3 = gaze3 * m
	vel = gaze3[:,:,-1] - gaze3[:,:,-2]

	### Make prediction, calculate loss. ###
	pred = gaze3[:, :, -1] + vel
	pred, _, _ = mask(pred, pred)
	pred_mask, label_mask, num = mask(pred, label)
	loss = torch.nn.MSELoss(size_average = False)(pred_mask, label_mask) / num
	return loss

def main():
	### Get the current datetime for naming. ###
	now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
	suffix = suffix='Itr: %(index)d of %(max)d. Avg: %(avg).02f/itr. Total: %(elapsed).02f. Remaining: %(eta).02f.'

	### Argument parsing. ###
	parser = argparse.ArgumentParser(description='Train a Multiscale Gazenet Model')

	### Miscellaneous options. ###
	parser.add_argument('-v', '--verbose', action='store_true')
	parser.add_argument('-w', '--workers', default=2, type=int)

	### Directory options. ###
	parser.add_argument('-tf', '--trial_file', default='config/good_gaze.txt')
	parser.add_argument('-d', '--data_dir', default='/home/ben/Desktop/harplabstuff/harpdata/gaze_tensors2')
	parser.add_argument('-lc', '--config_file', default=None)
	parser.add_argument('-s', '--save_dir', default=os.path.join(os.getcwd(), 'runs', now + '_' + socket.gethostname()))
	parser.add_argument('-c', '--save_config', default='config.pkl')
	parser.add_argument('-p', '--pred_dir', default='preds')
	parser.add_argument('-l', '--log_dir', default='logs')
	parser.add_argument('-m', '--model_dir', default='models')

	### Model options ###
	parser.add_argument('-w1', '--window1', default=12, type=int)
	parser.add_argument('-w2', '--window2', default=24, type=int)
	parser.add_argument('-w3', '--window3', default=36, type=int)

	### Training options. ###
	parser.add_argument('-train', '--train_pct', default=.9, type=float)
	parser.add_argument('-valid', '--valid_pct', default=.1, type=float)	
	parser.add_argument('-bs', '--batch_size', default=64, type=int)
	parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
	parser.add_argument('-e', '--epochs', default=100, type=int)

	args = parser.parse_args()

	if args.config_file and args._get_kwargs() == 1:
		### If we want to load from a config file, do so. ###
		if args.verbose:
			print(message('Loading config file.'))
		with open(args.config_file, 'rb') as f:
			args = pickle.load(f)
	elif args.config_file and args._get_kwargs() > 1:
		### If we have specified a config file and positional arguments ###
		### raise an exception.											###
		raise TypeError('train_gazenet.py takes only 1 positional argument when config_file is specified.')
	else:
		### Save the current configuration to a file in order to load later ###
		if args.verbose:
			print(message('Saving config file.'))
		os.system('mkdir -p ' + args.save_dir)
		os.system('touch ' + os.path.join(args.save_dir, args.save_config)) 
		with open(os.path.join(args.save_dir, args.save_config), 'ab') as f:
			pickle.dump(args, f)

	### Set some additional configuration arguments that are defined ###
	### from those the user gave through standard in.				 ###
	args.windows = [args.window3 - args.window1, args.window3 - args.window2, args.window3 - args.window3]
	args.test_pct = 1 - args.train_pct
	
	### Make all of the given directories or files absolute. ###
	args.save_config = os.path.join(args.save_dir, args.save_config)
	args.pred_dir = os.path.join(args.save_dir, args.pred_dir)
	args.log_dir = os.path.join(args.save_dir, args.log_dir)
	args.model_dir = os.path.join(args.save_dir, args.model_dir)

	### Create the above directories. ###
	os.system('mkdir -p ' + args.pred_dir)
	os.system('mkdir -p ' + args.log_dir)
	os.system('mkdir -p ' + args.model_dir)

	### Create a Tensorboard SummaryWriter. ###
	if args.verbose:
		print(message('Creating SummaryWriter.'))

	with SummaryWriter() as writer:
		### Create the necessary datasets and dataloaders. ###
		if args.verbose:
			print(message('Creating datasets and dataloaders.'))
		#trials, types = utils.get_trials(args.trial_file)
		#trials = utils.prefix_to_list(args.data_dir, trials)
		#total, train, valid, test  = utils.test_train_split(GazeDataset, args.train_pct, valid=args.valid_pct, trials=trials, windows=args.windows)
		
		#train_trials, train_types, valid_trials, valid_types, test_trials, test_types = split_inds(args.trial_file, args.train_pct, valid=args.valid_pct, prefix=args.data_dir)
		
		train_trials = utils.prefix_to_list(args.data_dir, utils.load_pickle('config/train_trials.pkl'))
		valid_trials = utils.prefix_to_list(args.data_dir, utils.load_pickle('config/valid_trials.pkl'))
		test_trials = utils.prefix_to_list(args.data_dir, utils.load_pickle('config/test_trials.pkl'))

		train = GazeDataset(train_trials, args.windows)
		valid = GazeDataset(valid_trials, args.windows)
		test = GazeDataset(test_trials, args.windows)

		trainloader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
		validloader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

		### Save the train, valid, and test inds. ###
		if args.verbose:
			print(message('Saving index orders.'))
		utils.save_inds(train.inds, os.path.join(args.save_dir, 'train_inds.txt'))
		utils.save_inds(valid.inds, os.path.join(args.save_dir, 'valid_inds.txt'))
		utils.save_inds(test.inds, os.path.join(args.save_dir, 'test_inds.txt'))

		### Create the model, criteria, and optimizer ###
		if args.verbose:
			print(message('Creating model and optimizer.'))
		model = GazeEncoderMS(args.window1, args.window2, args.window3)
		criterion = nn.MSELoss(size_average=False)
		optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

		# dummy_input = (Variable(torch.rand(64,2,12)), Variable(torch.rand(64,2,24)), Variable(torch.rand(64,2,36)), )
		# writer.add_graph(model, dummy_input, verbose=True)

		### Set some incrementers and begin training. ###
		if args.verbose:
			print(message('Starting to train.'))
		min_valid_loss = 999999999999999
		num_train_iter = 0
		num_valid_iter = 0
		epoch = 0
		for epoch in range(args.epochs):
			if args.verbose:
				print(message('Starting Epoch {:d}.'.format(epoch)))

			model_name = 'model_{:05d}.model'.format(epoch+1)

			### Specify where train pand valid predictions should be saved ###
			epoch_train_dir = os.path.join(args.save_dir, args.pred_dir, 'epoch_{:05d}'.format(epoch+1), 'train')
			epoch_valid_dir = os.path.join(args.save_dir, args.pred_dir, 'epoch_{:05d}'.format(epoch+1), 'valid')
			os.system('mkdir -p ' + epoch_train_dir)
			os.system('mkdir -p ' + epoch_valid_dir)

			### Begin the training loop. ###
			train_bar = bar.Bar(message('Training'), max=len(train) / args.batch_size, suffix=suffix)
			train_loss = 0.
			train_copy_loss = 0.
			train_vel_loss = 0.
			if args.verbose:
				print(message('Enumerating Trainloader.'))
			for i, data in enumerate(trainloader, 1):
				train_bar.next()

				num_train_iter += 1

				optimizer.zero_grad()

				### Get the data and turn them into Variables. ###
				gaze1 = Variable(data[:,2:,args.windows[0]:args.window3])
				gaze2 = Variable(data[:,2:,args.windows[1]:args.window3])
				gaze3 = Variable(data[:,2:,args.windows[2]:args.window3])
				label = Variable(data[:,2:,-1])

				### Forward pass. ###
				pred = model(gaze1, gaze2, gaze3)

				### Calculate loss. ###
				pred_mask, label_mask, num = mask(pred, label)
				loss = criterion(pred_mask, label_mask) / num
				train_loss += loss.data[0]

				### Calculate some other losses ###
				c_loss = copy_loss(gaze3, label)
				v_loss = const_vel_loss(gaze3, label)
				train_copy_loss += c_loss.data[0]
				train_vel_loss += v_loss.data[0]

				### Backward pass. ###
				loss.backward()

				### Optimize. ###
				optimizer.step()
				
				### Save pertinent losses to tensorboard. ###
				for b in range(args.batch_size):
					num = (num_train_iter-1) * args.batch_size + b
					writer.add_scalar('train/data/true_x', label.data[b,0], num)
					writer.add_scalar('train/data/true_y', label.data[b,1], num)
					writer.add_scalar('train/data/pred_x', pred.data[b,0], num)
					writer.add_scalar('train/data/pred_y', pred.data[b,1], num)
					writer.add_scalar('train/data/mask/true_x', label_mask.data[b,0], num)
					writer.add_scalar('train/data/mask/true_y', label_mask.data[b,1], num)
					writer.add_scalar('train/data/mask/pred_x', pred_mask.data[b,0], num)
					writer.add_scalar('train/data/mask/pred_y', pred_mask.data[b,1], num)

				writer.add_scalar('train/losses/train_loss_itr', loss.data[0], num_train_iter)
				writer.add_scalar('train/losses/train_copyloss_itr', c_loss.data[0], num_train_iter)
				writer.add_scalar('train/losses/train_constvelloss_itr', v_loss.data[0], num_train_iter)

				for name, param in model.named_parameters():
					writer.add_histogram(name, param.clone().data.numpy(), num_train_iter)

				torch.save(pred.data, os.path.join(epoch_train_dir, 'pred_{:05d}.pt7'.format(i+1)))

			train_bar.finish()

			### Add the average loss over the epoch to the tensorboard. ###
			avg_train_loss = (train_loss / len(train)) * args.batch_size
			avg_c_loss = (train_copy_loss / len(train)) * args.batch_size
			avg_v_loss = (train_vel_loss / len(train)) * args.batch_size
			writer.add_scalar('train/losses/train_loss_epoch', avg_train_loss, epoch)
			writer.add_scalar('train/losses/train_copyloss_epoch', avg_c_loss, epoch)
			writer.add_scalar('train/losses/train_constvelloss_epoch', avg_v_loss, epoch)

			### Begin validation loop. ###
			valid_bar = bar.Bar(message('Validate'), max=len(valid), suffix=suffix)
			valid_loss = 0
			valid_copy_loss = 0.
			valid_vel_loss = 0.
			for i, data in enumerate(validloader):
				num_valid_iter += 1

				valid_bar.next()

				### Get the data and turn them into Variables. ###
				gaze1 = Variable(data[:,2:,:args.windows[0]])
				gaze2 = Variable(data[:,2:,:args.windows[1]])
				gaze3 = Variable(data[:,2:,:args.windows[2]])
				label = Variable(data[:,2:,-1])

				### Forward pass. ###
				pred = model(gaze1, gaze2, gaze3)

				### Calculate loss. ###
				pred_mask, label_mask, num = mask(pred, label)
				loss = criterion(pred_mask, label_mask) / num
				valid_loss += loss.data[0]
				
				c_loss = copy_loss(gaze3, label)
				valid_copy_loss += c_loss.data[0]

				cv_loss = const_vel_loss(gaze3, label)
				valid_vel_loss += cv_loss.data[0]

				### Save pertinent losses to tensorboard. ###
				writer.add_scalar('valid/data/true_x', label.data[0,0], num_valid_iter)
				writer.add_scalar('valid/data/true_y', label.data[0,1], num_valid_iter)
				writer.add_scalar('valid/data/pred_x', pred.data[0,0], num_valid_iter)
				writer.add_scalar('valid/data/pred_y', pred.data[0,1], num_valid_iter)
				writer.add_scalar('valid/data/mask/true_x', label_mask.data[0,0], num_valid_iter)
				writer.add_scalar('valid/data/mask/true_y', label_mask.data[0,1], num_valid_iter)
				writer.add_scalar('valid/data/mask/pred_x', pred_mask.data[0,0], num_valid_iter)
				writer.add_scalar('valid/data/mask/pred_y', pred_mask.data[0,1], num_valid_iter)
				
				writer.add_scalar('valid/losses/valid_loss', loss.data[0], num_valid_iter)
				writer.add_scalar('valid/losses/valid_copyloss', c_loss.data[0], num_valid_iter)
				writer.add_scalar('valid/losses/valid_constvelloss', cv_loss.data[0], num_valid_iter)

				torch.save(pred.data, os.path.join(epoch_valid_dir, 'pred_{:05d}.pt7'.format(i+1)))
				
			valid_bar.finish()

			### Add the average loss over the epoch to the tensorboard. ###
			avg_valid_loss = valid_loss / len(valid) * args.batch_size
			writer.add_scalar('valid/losses/valid_loss_epoch', avg_valid_loss, epoch)

			avg_copy_loss = valid_copy_loss / len(valid) * args.batch_size
			writer.add_scalar('valid/losses/valid_copyloss_epoch', avg_copy_loss, epoch)

			avg_vel_loss = valid_vel_loss / len(valid) * args.batch_size
			writer.add_scalar('valid/losses/valid_constvelloss_epoch', avg_vel_loss, epoch)

			### Export all scalars to json for later use. ###
			if epoch % 5 == 0 :
				writer.export_scalars_to_json(os.path.join(args.log_dir, 'log.json'))

			### Save the model if the current epoch's validation loss was less than ###
			### the previous minimum.												###
			if avg_valid_loss < min_valid_loss:
				min_valid_loss = avg_valid_loss
				optimizer.zero_grad()
				torch.save(model.state_dict(), os.path.join(args.model_dir, model_name))

if __name__ == '__main__':
	main()









