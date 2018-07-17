import numpy as np

import os
import os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skimage
import skimage.io as skio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from EgoEncoder import *
from GazeData import *
from GazeMapper import *
from average import *
from config import *

from datetime import datetime


#PATH = '/media/ben/HARPLab-2T1/eyegaze_data_ada_eating_study/cleaned_data'
PATH = '/mnt/sdb1/bnewman1/harpdata/cleaned_data'
GAZE_NAME = 'gaze_positions.csv'
FEATURE_NAME = 'features'
FRAME_NAME = 'frames'
WINDOW_SIZE = 36
BATCH_SIZE = 25
IM_HEIGHT = 224
IM_WIDTH = 224
IN_CHANNELS = 4
LOAD_FROM_SAVE = False
checkpoint = '/mnt/sdb1/bnewman1/harpmodels/egonet_2017-12-05_23:48:30.t7'

def save_image(im, mask):
	print(im.dtype)
	tmp_im = np.zeros((224,224,3))
	tmp_im[:,:,0] = np.multiply(im[:,:,0], mask)
	tmp_im[:,:,1] = np.multiply(im[:,:,1], mask)
	tmp_im[:,:,2] = np.multiply(im[:,:,2], mask)
	
	print(np.max(mask), np.max(im[:,:,0]), np.max(im[:,:,1]), np.max(im[:,:,2]), np.max(tmp_im[:,:,0]), np.max(tmp_im[:,:,1]), np.max(tmp_im[:,:,2]))
	plt.imshow(tmp_im.astype('uint8'))
	plt.savefig('./test_ims/test.png')
	plt.close()

	plt.imshow(im)
	plt.savefig('./test_ims/test_im.png')
	plt.close()

def load_batches(gd, gm, inds, bsz, pnum, tnum, tot):
	for i in range(len(inds)):
		batch = np.zeros((bsz, IM_WIDTH, IM_HEIGHT, IN_CHANNELS)).astype('uint8')
		for j in range(bsz):
			print(inds[i]+j, tot)
			if (inds[i]+j) >= np.ceil(tot):
				print('batches HELLLLOOOOOOO', tot, (inds[i]+j)/25, np.ceil(tot/25.))
				break
			pts = gd[inds[i]+j:inds[i]+j+WINDOW_SIZE, 1:]
			mask = gm.create_masks(pts)

			ex = np.zeros((IM_WIDTH, IM_HEIGHT, IN_CHANNELS)).astype('uint8')
			vid = VID_PARTS[int(pnum[1:])-1]
			im = skio.imread(os.path.join(vid, tnum, 'world_{:05d}.png'.format(inds[i]+j)))
			
			#save_image(im, mask)
			
			ex[:,:,0:3] = im
			ex[:,:,3] = (mask*255).astype('uint8')
			batch[j] = ex
		yield torch.from_numpy(batch).view(25,4,224,224)

def load_labels(inds, bsz, pnum, tnum, tot):
	for i in range(len(inds)):
		features = torch.zeros(bsz, 4096)
		for j in range(bsz):
			feat = FEAT_PARTS[int(pnum[1:])-1]
			feat_num =  (inds[i]+j)/25.
			feat_ind = int(np.round((feat_num - int(feat_num)) * 25))
			num_files = len(os.listdir(os.path.join(feat, tnum)))
			if int(np.round(feat_num)+1) > np.ceil(tot/25.):
				print('LAbels HELLLLOOOOOOO', tot, np.ceil(tot/25.))
				break
			tmp = torch.load(os.path.join(feat, tnum, '{:05d}.t7'.format(int(np.round(feat_num)+1))))
			tmp_feat = torch.zeros(WINDOW_SIZE, 4096)
			offset = 0
			for k in range(WINDOW_SIZE):
				if feat_ind+k > num_files or feat_ind+k-offset > tmp.data.shape[0]:
					pass
				elif feat_ind+k < 25:
					offset = 0
				elif feat_ind+k >= 25 and feat_ind+k < 50 and int(feat_num)+2 < num_files:
					offset = 25
					tmp = torch.load(os.path.join(feat, tnum, '{:05d}.t7'.format(int(feat_num)+2)))
				elif feat_ind+k >=50 and feat_ind+k < 75 and int(feat_num)+3  < num_files:
					offset = 50
					tmp = torch.load(os.path.join(feat, tnum, '{:05d}.t7'.format(int(feat_num)+3)))
				else:
					pass
				tmp_feat[k] = tmp.data[(feat_ind+k)%25]
			features[j] = torch.sum(tmp_feat,0) / tmp_feat.shape[0]
		yield features

egonet = None
loss_file = '/mnt/sdb1/bnewman1/harplogs/egonet_loss.txt'
lowest_loss = 1000000000000
loss_log = None
if LOAD_FROM_SAVE:
	egonet = EgoEncoder('vgg13', False)
	params = torch.load(checkpoint)
	egonet.load_state_dict(params)
	loss_log = open(loss_file, 'a+')
	lowest_loss = float(loss_log.read().split('\n')[-2])
	loss_log.seek(0)
else:
	egonet = EgoEncoder('vgg13', True)
	loss_log = open(loss_file, 'w')


criterion = nn.MSELoss()
egonet.cuda()
criterion.cuda()

optimizer = optim.Adam(egonet.parameters(), lr=0.001)

gd = GazeData()
gm = GazeMapper()

parts = [os.path.join(PATH, part) for part in os.listdir(PATH) if len(part.split('.')) == 1]
trials = []
for i in range(len(parts)):
	tmp_trial = [os.path.join(parts[i],trial) for trial in os.listdir(parts[i])]
	trials += tmp_trial

running_loss = 0.0
while True:
	order = np.random.permutation(trials)
	for trial in order:
		tnum = trial.split('/')[-1]
		pnum = trial.split('/')[-2]

		gd.load_csv(os.path.join(trial, GAZE_NAME))
		vid = VID_PARTS[int(pnum[1:])-1]
		num_frames = len(os.listdir(os.path.join(vid, tnum)))
		print(os.path.join(vid, tnum))
		print(num_frames)

		order = np.random.permutation(range(1, num_frames+1, 25))
		batches = iter(load_batches(gd.data, gm, order, BATCH_SIZE, pnum, tnum, num_frames))
		labels = iter(load_labels(order, BATCH_SIZE, pnum, tnum, num_frames))

		for i in range(len(order)):
			try:
				inp = Variable(batches.next()).float()
				lbl = Variable(labels.next()).float()
			except StopIteration:
				pass
			inp = inp.cuda()
			lbl = lbl.cuda()

			optimizer.zero_grad()
			print('forward')
			out = egonet(inp)
			print('loss')
			loss = criterion(out, lbl)
			print('backward')
			loss.backward()
			print('step')
			optimizer.step()
			print(loss.data[0])
			if loss.data[0] < lowest_loss:
				lowest_loss = loss.data[0]
				dt = datetime.now().strftime('egonet_%Y-%m-%d_%H:%M:%S')
				torch.save(egonet.state_dict(), '/mnt/sdb1/bnewman1/harpmodels/'+dt+'.t7')
			loss_log.write(str(loss.data[0])+'\n')
			running_loss += loss.data[0]
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0







