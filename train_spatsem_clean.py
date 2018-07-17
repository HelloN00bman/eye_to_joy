import numpy as np
import csv
import os
import os.path
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from torch.autograd import Variable
from config import *
from SpatSemLSTM import *
from spatsemdataset import *
from torch.utils.data import *

#data_path = '/mnt/sdb1/bnewman1/harpdata/'
#gaze_path = 'cleaned_data/'
#joy_path = 'cleaned_data/'
#video_path = 'eyegaze_videos/'
#gaze_name = 'gaze_positions.csv'
#joy_name = 'traj_info.csv'
#video_name = 'world'
windows = data_dict['WINDOWS']
window = data_dict['WINDOW_SIZE']
num_workers = 2
epochs = 2


#print 'preprocessing'

trials = []
types = []
with open('good_gaze.txt') as f:
	lines = f.readlines()
	for line in lines:
		tmp = line.strip().split('.')
		trials+=[os.path.join(tmp[0].split("'")[-1], tmp[1].split(' ')[0])]
		types+=[tmp[1].split(' ')[1][1]]

trials = np.array(trials)
types = np.array(types)
trials = trials[np.where(types != 'a')[0]]

#print 'data loading'
dataset = SpatSemDataset(data_dict)
#dataloader = DataLoader(dataset, shuffle=True, num_workers=0)


#print 'model loading'
model = SpatSemLSTM().cuda()

#print 'loss loading'
criterion = nn.MSELoss().cuda()

#print 'optimizer loading'
optimizer = optim.Adam(model.parameters(), lr=0.001)
#trainer = SpatSemTrainer(model, criterion, optimize, dataloader)

#print 'starting loop'
for i in range(1,epochs+1):
	#print 'epoch' , i, 'start'
	
	#for j, data_all in enumerate(dataloader, 1):
	
	gaze_all, ego_all, label_all = dataset[int(sys.argv[1])]
	#print 'sequence', 1, 'out of 1'

	sem_out = Variable(torch.zeros(1,1,4096))
	sem_hidden = (Variable(torch.zeros(1, 1, 4096).float()), Variable(torch.zeros(1, 1, 4096).float()))##model.init_hidden(4096)
	spat_hidden = (Variable(torch.zeros(1, 1, 4096).float()), Variable(torch.zeros(1, 1, 4096).float()))#model.init_hidden(4096)
	
	for k in range(1):#range(0,gaze_all.shape[1],108):
		#print 'chunk', (k)/108 + 1, 'out of', gaze_all.shape[1]
		optimizer.zero_grad()
		loss_tot = 0
		sem_out = Variable(sem_out.data.cuda())
		sem_hidden = (Variable(sem_hidden[0].data.cuda()), Variable(sem_hidden[1].data.cuda()))
		spat_hidden = (Variable(spat_hidden[0].data.cuda()), Variable(spat_hidden[1].data.cuda()))

		for l in range(gaze_all.shape[0]):
			#print 'itr', l+1, 'out of',gaze_all.shape[0]

			gaze = gaze_all[l,:,:]
			ego = ego_all[(l)/36,:,:,:]
			label = label_all[l,:]
			
			gaze1 = Variable(gaze[window - windows[0]:, :].view(1, 2, -1).cuda())
			gaze2 = Variable(gaze[window - windows[1]:, :].view(1, 2, -1).cuda())
			gaze3 = Variable(gaze[window - windows[2]:, :].view(1, 2, -1).cuda())
			ego = Variable(ego.view(1, 4, 224, 224).cuda())
			label = Variable(label.view(1, 2).cuda())

			try:
				out, sem_out, sem_hidden, spat_hidden = model(l, ego, gaze1, gaze2, gaze3, sem_out, sem_hidden, spat_hidden)
			except RuntimeError as e:
				f = open('errors.txt', 'wb')
				f.write(sys.argv[1]+'\n')
				f.write('forward error\n')
				f.write(str(e)+'\n')
				f.close()
				print(sys.argv[1])
				print(e)
			try:
				loss_tot += criterion(out, label)
			except RuntimeError as e:
				f = open('errors.txt', 'wb')
				f.write(sys.argv[1]+'\n')
				f.write('loss error\n')
				f.write(str(e)+'\n')
				f.close()
				print(sys.argv[1])
				print(e)
			#print out, label
			#print loss_tot.data[0]

		#print loss_tot.data[0]
		#print 'backward'
		try:
			loss_tot.backward()
		except RuntimeError as e:
			f = open('errors.txt', 'wb')
			f.write(sys.argv[1]+'\n')
			f.write('backward error')
			f.write(str(e)+'\n')
			f.close()
			print(sys.argv[1])
			print(e)
		nn.utils.clip_grad_norm(model.parameters(), 0.25)
		optimizer.step()
		torch.cuda.empty_cache()
		













