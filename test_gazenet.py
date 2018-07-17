import argparse
import csv
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
	parser.add_argument('-tf', '--trial_file', default='config/test_trials.pkl')
	parser.add_argument('-tyf', '--type_file', default='config/test_types.pkl')
	parser.add_argument('-d', '--data_dir', default='/home/ben/Desktop/harplabstuff/harpdata/gaze_tensors2')
	parser.add_argument('-s', '--save_dir', default=os.path.join(os.getcwd(), 'runs', now + '_' + socket.gethostname()))
	parser.add_argument('-c', '--save_config', default='config.pkl')
	parser.add_argument('-p', '--pred_dir', default='preds')
	parser.add_argument('-l', '--log_dir', default='logs')
	parser.add_argument('-m', '--model_dir', default='models')

	### Model options ###
	parser.add_argument('-w1', '--window1', default=12, type=int)
	parser.add_argument('-w2', '--window2', default=24, type=int)
	parser.add_argument('-w3', '--window3', default=36, type=int)

	### Testing Options ### 
	parser.add_argument('-bs', '--batch_size', default=64, type=int)
	parser.add_argument('-lm', '--load_model', default='/home/ben/Desktop/harplabstuff/harpcode/eye_to_joy/runs/Jan25_01-57-27_Aeolus/models/model_00064.model')
	#parser.add_argument('-', '--window1', default=12, type=int)

	args = parser.parse_args()

	"""
	if args.config_file and args._get_kwargs() == 1:
		### If we want to load from a config file, do so. ###
		if args.verbose:
			print message('Loading config file.')
		with open(args.config_file, 'rb') as f:
			args = pickle.load(f)
	elif args.config_file and args._get_kwargs() > 1:
		### If we have specified a config file and positional arguments ###
		### raise an exception.											###
		raise TypeError('train_gazenet.py takes only 1 positional argument when config_file is specified.')
	else:
		### Save the current configuration to a file in order to load later ###
		if args.verbose:
			print message('Saving config file.')
		os.system('mkdir -p ' + args.save_dir)
		os.system('touch ' + os.path.join(args.save_dir, args.save_config)) 
		with open(os.path.join(args.save_dir, args.save_config), 'ab') as f:
			pickle.dump(args, f)
	"""
	args.windows = [args.window1, args.window2, args.window3]

	args.save_config = os.path.join(args.save_dir, args.save_config)
	args.pred_dir = os.path.join(args.save_dir, args.pred_dir)
	args.log_dir = os.path.join(args.save_dir, args.log_dir)
	args.model_dir = os.path.join(args.save_dir, args.model_dir)

	test_trials = utils.prefix_to_list(args.data_dir, utils.load_pickle(args.trial_file))	
	test_types = utils.load_pickle(args.type_file)
	print test_types

	test = GazeDataset(test_trials, args.windows, test_types)

	testloader = DataLoader(test, batch_size=1, shuffle=False, drop_last=False, num_workers=args.workers)
	lens = test._trial_lengths
	tps = test.types
	print(tps)

	model = GazeEncoderMS(args.window1, args.window2, args.window3, eval=True)
	model.load_state_dict(torch.load(args.load_model))
	model = model.eval()

	with open('test_results.csv', 'ab') as csvfile:
		csvwriter = csv.writer(csvfile, delimiter=',')
		tot = 0
		test_bar = bar.Bar(message('Testing'), max=len(test) / 1)
		for i, data in enumerate(testloader, 1):
			tsum = 0
			frames = []
			vids = []
			totlens = []
			types = []
			for j in range(len(lens)):
				if i+36*j - (lens[j][1]+tsum) <= 0:
					frames.append(i+36*j-tsum)
					vids.append(lens[j][0])
					totlens.append(lens[j][1])
					types.append(tps[j])
					break
				else:
					tsum+=lens[j][1]

			## NEED TO CHANGE THIS
			gaze1 = Variable(data[:,2:,:args.windows[0]])
			gaze2 = Variable(data[:,2:,:args.windows[1]])
			gaze3 = Variable(data[:,2:,:args.windows[2]])
			label = Variable(data[:,2:,-1])

			pred = model(gaze1, gaze2, gaze3)

			pred_mask, label_mask, num = mask(pred, label)

			tot+=torch.sum(pred_mask - label_mask)/num

			csvwriter.writerow(frames+vids+totlens+types+list(pred.data[0])+list(label.data[0]))
			test_bar.next()
		test_bar.finish()
		print(tot/i)

if __name__ == '__main__':
	main()