import numpy as np

import os
import os.path

import skimage
import skimage.io as skio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag

from EgoEncoder import *
from GazeEncoderMS import *
from GazeData import *
from GazeMapper import *
from average import *
from config import *

from datetime import datetime

PATH = '/mnt/sdb1/bnewman1/harpdata/cleaned_data'
GAZE_NAME = 'gaze_positions.csv'
FEATURE_NAME = 'features'
FRAME_NAME = 'frames'
WINDOWS = [12, 24, 36]
BATCH_SIZE = 25
IM_HEIGHT = 224
IM_WIDTH = 224
IN_CHANNELS = 2
LOAD_FROM_SAVE = False
checkpoint = '/mnt/sdb1/bnewman1/harpmodels/gazenet_2017-12-05_23:48:30.t7'


def load_batches(inds, bsz, gd):
	# This should yield  ([b1, b2, b3], l) where:
	#	 b1 is a batch of size bsz x IN_CHANNELS x WINDOWS[0]
	#	 b2 is a batch of size bsz x IN_CHANNELS x WINDOWS[1]
	#    b3 is a batch of size bsz x IN_CHANNELS x WINDOWS[2]
	#    l is a label of size bsz x 
	tot = min(len(inds), gd.shape[0])
	for i in range(tot):
		b1 = torch.zeros(bsz, IN_CHANNELS, WINDOWS[0])
		b2 = torch.zeros(bsz, IN_CHANNELS, WINDOWS[1])
		b3 = torch.zeros(bsz, IN_CHANNELS, WINDOWS[2])
		l = torch.zeros(bsz, IN_CHANNELS)
		for j in range(bsz):
			if i+j >= len(inds):
				break
			ind = inds[i+j]
			end = ind + 1
			if end >= tot or end + 1 >= tot:
				break
			b1[j,:,:] = torch.from_numpy(gd[ind-WINDOWS[0]+1:end, 1:])
			b2[j,:,:] = torch.from_numpy(gd[ind-WINDOWS[1]+1:end, 1:])
			b3[j,:,:] = torch.from_numpy(gd[ind-WINDOWS[2]+1:end, 1:])
			l[j] = torch.from_numpy(gd[end+1,1:])
		yield (b1,b2,b3,l)

gazenet = GazeEncoderMS(*WINDOWS)
loss_file = '/mnt/sdb1/bnewman1/harplogs/gazenet_loss.txt'
lowest_loss = 1000000000000
loss_log = None
if LOAD_FROM_SAVE:
	params = torch.load(checkpoint)
	gazenet.load_state_dict(params)
	loss_log = open(loss_file, 'a+')
	lowest_loss = float(loss_log.read().split('\n')[-2])
	loss_log.seek(0)
else:
	loss_log = open(loss_file, 'w')

criterion = nn.MSELoss()
optimizer = optim.Adam(gazenet.parameters(), lr=0.001)
gd = GazeData()

parts = [os.path.join(PATH, part) for part in os.listdir(PATH) if len(part.split('.')) == 1]
trials = []
for i in range(len(parts)):
	tmp_trial = [os.path.join(parts[i],trial) for trial in os.listdir(parts[i])]
	trials += tmp_trial

running_loss = 0
while True:
	trial_order = np.random.permutation(trials)
	for itr in trial_order:
		gd.load_csv(os.path.join(itr, GAZE_NAME))

		inds = range(max(WINDOWS), len(gd.data)-1, 1)
		order = np.random.permutation(inds)
		batches = load_batches(order, BATCH_SIZE, gd.data)
		for i in order:
			b1, b2, b3, lbl = batches.next()
			b1 = ag.Variable(b1)
			b2 = ag.Variable(b2)
			b3 = ag.Variable(b3)
			lbl = ag.Variable(lbl)

			optimizer.zero_grad()
			print('forward')
			out = gazenet(b1, b2, b3)
			print('loss')
			loss = criterion(out, lbl)
			print('backward')
			loss.backward()
			print('optimise')
			optimizer.step()
			print(loss.data[0])
			if loss.data[0] < lowest_loss:
				lowest_loss = loss.data[0]
				dt = datetime.now().strftime('gazenet_%Y-%m-%d_%H:%M:%S')
				torch.save(gazenet.state_dict(), '/mnt/sdb1/bnewman1/harpmodels/'+dt+'.t7')
			loss_log.write(str(loss.data[0])+'\n')
			running_loss += loss.data[0]
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0











	