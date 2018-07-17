import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import os.path
import scipy.signal as signal
import skimage
import skimage.io as skio
import torch
import time
import gc

from torch.utils.data import *
from eye_to_joy.utils.GazeMapper import *

class SpatSemDataset(Dataset):
	def __init__(self, data_dict):
		super(SpatSemDataset, self).__init__()
		self.data_dict = data_dict

		self.h = self.data_dict['IM_HEIGHT']
		self.w = self.data_dict['IM_WIDTH']
		self.px = 255

		self.gaze_paths = self.data_dict['GAZE_TRIALS']
		self.joy_paths = self.data_dict['JOY_TRIALS']
		self.vid_paths = self.data_dict['VID_TRIALS']
		
		self.gaze_name = self.data_dict['GAZE_NAME'] 
		self.joy_name = self.data_dict['JOY_NAME']
		self.video_name = self.data_dict['VID_NAME']
		
		self.trials = self.data_dict['TRIALS']
		self.window_size = self.data_dict['WINDOW_SIZE']
		self.windows = self.data_dict['WINDOWS']

		self.GazeMapper = GazeMapper()

	def __getitem__(self, index):
		gaze_path = self.gaze_paths[index]
		joy_path = self.joy_paths[index]
		vid_path = self.vid_paths[index]

		gaze_x, gaze_y, gaze_len = self.process_gaze(gaze_path)
		joy_x, joy_y, joy_len = self.process_joy(joy_path, gaze_len)

		frames = os.listdir(vid_path)
		frames.sort()
		video_len = len(frames)
		
		start = self.window_size
		end = gaze_len if gaze_len < video_len else video_len
		end = (end / start)*start

		gaze_all = torch.zeros(end-start, self.window_size, 2)
		ego_all = torch.zeros((end-start)/start, 4, self.w, self.h)
		labels_all = torch.zeros(end-start, 2)

		for i in range(start, end, 1):
			entry_ind = i - start
			labels_all[entry_ind,0] = joy_x[i]
			labels_all[entry_ind,1] = joy_y[i]
			
			gaze_all[entry_ind,:,0] = torch.from_numpy(gaze_x[i-self.windows[2]+1:i+1])
			gaze_all[entry_ind,:,1] = torch.from_numpy(gaze_y[i-self.windows[2]+1:i+1])
			
			gaze = np.array((gaze_x[i-self.windows[2]+1:i+1], gaze_y[i-self.windows[2]+1:i+1])).T
			if not(i%start):
				frame = np.zeros((4, self.w, self.h)).astype('uint8')
				gaze_map = self.GazeMapper.create_masks(gaze)
				frame_path = os.path.join(vid_path, frames[i+start-1])
				im = skio.imread(frame_path).reshape(3,self.w, self.h)
				frame[0:3, :, :] = im
				frame[3, :, :] = gaze_map * self.px
				frame = torch.from_numpy(frame)
				ego_all[entry_ind/start] = frame

		return gaze_all, ego_all, labels_all

	def __len__(self):
		return len(self.trials)

	def process_gaze(self, gaze_path, fill = -1):
		gaze_path = os.path.join(gaze_path, self.gaze_name)
		gaze_header, gaze_data = self.load_csv(gaze_path, True)
		gaze_len = gaze_data.shape[0]
		
		gaze_x = (np.array(gaze_data[:,4]).astype('float64') * self.w).astype('int64')
		gaze_y = (np.array(gaze_data[:,5]).astype('float64') * self.h).astype('int64')
		
		too_big_x = np.where(gaze_x >= self.w)
		too_small_x = np.where(gaze_x < 0)
		too_big_y = np.where(gaze_y >= self.h)
		too_small_y = np.where(gaze_y < 0)
		
		gaze_x[too_big_x] = fill
		gaze_x[too_big_y] = fill
		gaze_x[too_small_x] = fill
		gaze_x[too_small_y] = fill

		gaze_y[too_big_x] = fill
		gaze_y[too_big_y] = fill
		gaze_y[too_small_x] = fill
		gaze_y[too_small_y] = fill

		return gaze_x, gaze_y, gaze_len

	def process_joy(self, joy_path, gaze_len):
		joy_path = os.path.join(joy_path, self.joy_name)
		joy_header, joy_data = self.load_csv(joy_path, True)
		joy_len = joy_data.shape[0]
		
		joy_x = np.array(joy_data[:,2]).astype('float64')
		joy_y = np.array(joy_data[:,3]).astype('float64')

		assert(gaze_len <= joy_len)
		joy_data_rsmpl = self.resample(np.array((joy_x, joy_y)).T, gaze_len)
		joy_len = joy_data_rsmpl.shape[0]
		joy_x = joy_data_rsmpl[:,0]
		joy_y = joy_data_rsmpl[:,1]

		return joy_x, joy_y, joy_len

	def load_csv(self, csvfile, header):
		lines = []
		head = []
		data = []
		with open(csvfile) as f:
			reader = csv.reader(f)
			for line in reader:
				lines += [line]
		if header:
			head = np.array(lines[0])
			data = np.array(lines[1:])
		else:
			data = np.array(lines)
		return head, data

	def resample(self, seq, n):
		return signal.resample(seq, n)

	def save_hmaps(self):
		save_path = '/home/bnewman1/test_hmaps/'
		for i in range(len(self)):
			trial = self.trials[i]
			part = trial.split('/')[0]
			os.system('mkdir -p ' + save_path+trial)
			num = trial.split('/')[-1]
			gaze, ego, labels = self[i]
			j = 1
			for frame in ego:
				print(save_path+trial+'/hmap_'+str(j*36)+'.png')
				hmap = frame[3,:,:]
				plt.imshow(hmap)
				plt.title(part + ' ' + num)
				plt.savefig(save_path+trial+'/hmap_'+str(j*36)+'.png')
				plt.close()
				ego_im = frame[0:3,:,:].reshape(224,224,3)
				plt.imshow(ego_im)
				plt.title(part + ' ' + num)
				plt.savefig(save_path+trial+'/ego_'+str(j*36)+'.png')
				j+=1
			print('done with ', i, 'out of ', len(self))

