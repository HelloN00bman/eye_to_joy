import numpy as np

import os
import os.path

import skimage
import skimage.io as skio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from EgoEncoder import *
from NaiveMapper import *
from GazeEncoderMS import *
from GazeData import *
from GazeMapper import *
from average import *
from config import *

from datetime import datetime

PATH = '/mnt/sdb1/bnewman1/harpdata/cleaned_data'
GAZE_NAME = 'gaze_positions.csv'
JOY_NAME = 'traj_info.csv'
FEATURE_NAME = 'features'
FRAME_NAME = 'frames'
WINDOWS = [12, 24, 36]
WINDOW_SIZE = max(WINDOWS)
BATCH_SIZE = 1
IM_HEIGHT = 224
IM_WIDTH = 224
IN_CHANNELS = 2
LOAD_FROM_SAVE = False
checkpoint = '/mnt/sdb1/bnewman1/harpmodels/naivenet_2017-12-05_23:48:30.t7'

def load_batches(gd, jd, gm, order, BATCH_SIZE, pnum, tnum, num_frames):
	# i need an image, 36 gaze points

	# This should yield  ([b1, b2, b3], l) where:
	#	 b1 is a batch of size bsz x IN_CHANNELS x WINDOWS[0]
	#	 b2 is a batch of size bsz x IN_CHANNELS x WINDOWS[1]
	#    b3 is a batch of size bsz x IN_CHANNELS x WINDOWS[2]
	#    l is a label of size bsz x 
	tot = min(len(order), gd.shape[0])
	print(tot)
	bsz = BATCH_SIZE
	for i in range(tot):
		e = torch.from_numpy(np.zeros((bsz, 224, 224, 4)).astype('uint8'))
		b1 = torch.zeros(bsz, IN_CHANNELS, WINDOWS[0])
		b2 = torch.zeros(bsz, IN_CHANNELS, WINDOWS[1])
		b3 = torch.zeros(bsz, IN_CHANNELS, WINDOWS[2])
		l = torch.zeros(bsz, IN_CHANNELS)
		ind = order[i]
		if ind + WINDOW_SIZE + 1 >= tot:
			yield e.view(1,4,224,224),b1,b2,b3,l
		else:
			g_pts = gd[ind:ind+WINDOW_SIZE, 1:]
			mask = gm.create_masks(g_pts)

			vid = VID_PARTS[int(pnum[1:])-1]
			im = skio.imread(os.path.join(vid, tnum, 'world_{:05d}.png'.format(ind)))
			e[:,:,:,0:3] = torch.from_numpy(im)
			e[:,:,:,3] = torch.from_numpy((mask*255).astype('uint8'))
			e = e.view(1,4,224,224)
			b1[0,:,:] = torch.from_numpy(g_pts[WINDOW_SIZE - WINDOWS[0]:])
			b2[0,:,:] = torch.from_numpy(g_pts[WINDOW_SIZE - WINDOWS[1]:])
			b3[0,:,:] = torch.from_numpy(g_pts[WINDOW_SIZE - WINDOWS[2]:])

			l = torch.from_numpy(jd[ind+WINDOW_SIZE+1, 1:])

		yield (e,b1,b2,b3,l)

naivenet = None
loss_file = '/mnt/sdb1/bnewman1/harplogs/naivenet_loss.txt'
pred_file = '/mnt/sdb1/bnewman1/harplogs/naivenet_pred.txt'
lowest_loss = 1000000000000
loss_log = None
pred_log = None

if LOAD_FROM_SAVE:
	naivenet = NaiveMapper()
	params = torch.load(checkpoint)
	naivenet.load_state_dict(params)
	loss_log = open(loss_file, 'a+')
	lowest_loss = float(loss_log.read().split('\n')[-2])
	loss_log.seek(0)
	pred_log = open(pred_file, 'a+')
	pred_log.seek(0)
else:
	naivenet = NaiveMapper()
	loss_log = open(loss_file, 'w')
	pred_log = open(pred_file, 'w')


criterion = nn.MSELoss()
naivenet.cuda()
criterion.cuda()

optimizer = optim.Adam(naivenet.parameters(), lr=0.001)

gd = GazeData()
jd = GazeData(inds = [0,2,3], round=[0,0,0], w=2, h=2)
gm = GazeMapper()

parts = [os.path.join(PATH, part) for part in os.listdir(PATH) if len(part.split('.')) == 1]
trials = []
for i in range(len(parts)):
	tmp_trial = [os.path.join(parts[i],trial) for trial in os.listdir(parts[i])]
	trials += tmp_trial

running_loss = 0.0
epoch = 0
while True:
	order = np.random.permutation(trials)
	for trial in order:
		tnum = trial.split('/')[-1]
		pnum = trial.split('/')[-2]

		gd.load_csv(os.path.join(trial, GAZE_NAME))
		print(os.path.join(trial, JOY_NAME))
		jd.load_csv(os.path.join(trial, JOY_NAME))
		if jd.data.shape[0] == 0:
			print('no joy data')
			continue
		gd_len = gd.data.shape[0]
		jd_len = jd.data.shape[0]
		print(gd_len, jd_len)
		if gd_len > jd_len:
			gd.data = gd.data[:jd_len, :]
		else:
			jd.data = jd.data[:gd_len, :]
		gd_len = gd.data.shape[0]
		jd_len = jd.data.shape[0]
		print(gd_len, jd_len)

		vid = VID_PARTS[int(pnum[1:])-1]

		num_frames = len(os.listdir(os.path.join(vid, tnum)))
		print(os.path.join(vid, tnum))
		print(num_frames)

		order = np.random.permutation(range(1, num_frames+1))
		batches = iter(load_batches(gd.data, jd.data, gm, order, BATCH_SIZE, pnum, tnum, num_frames))
		#labels = iter(load_labels(order, BATCH_SIZE, pnum, tnum, num_frames))

		for i in range(len(order)):
			print('THIS IS I:', i)
			try:
				ego_batch, gaze_batch1, gaze_batch2, gaze_batch3, lbl = batches.next()
				ego_batch = Variable(ego_batch.float().cuda())
				gaze_batch1 = Variable(gaze_batch1.float().cuda())
				gaze_batch2 = Variable(gaze_batch2.float().cuda())
				gaze_batch3 = Variable(gaze_batch3.float().cuda())
				lbl = Variable(lbl.float().cuda())
			except StopIteration:
				print('stop iteration')
				pass
			"""
			ego_batch = ego_batch.cuda()
			gaze_batch1 = gaze_batch1.cuda()
			gaze_batch2 = gaze_batch2.cuda()
			gaze_batch3 = gaze_batch3.cuda()
			lbl = lbl.cuda()
			"""
			optimizer.zero_grad()
			naivenet.hidden = naivenet.init_hidden()
			print('forward')
			out = naivenet(ego_batch, gaze_batch1, gaze_batch2, gaze_batch3)
			print('loss')
			loss = criterion(out, lbl)
			print('backward')
			loss.backward()
			print('step')
			optimizer.step()
			print(loss.data[0])
			if loss.data[0] < lowest_loss:
				lowest_loss = loss.data[0]
				dt = datetime.now().strftime('naivenet_%Y-%m-%d_%H:%M:%S')
				torch.save(naivenet.state_dict(), '/mnt/sdb1/bnewman1/harpmodels/'+dt+'.t7')
			loss_log.write(str(loss.data[0])+'\n')
			#print out.data[0], lbl.data
			try:
				pred_log.write(str(out.data[0][0])+','+str(out.data[0][1])+','+str(lbl.data[0])+','+str(lbl.data[1])+'\n')
			except:
				print('do nothing')
			running_loss += loss.data[0]
			if i % 100 == 99:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 100))
				running_loss = 0.0
			epoch += 1


