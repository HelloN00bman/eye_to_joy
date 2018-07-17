import numpy as np
import os
import os.path
import torch

from skimage import io, transform
from torch.utils.data import *
from eye_to_joy.utils import *

class EgoDataset(Dataset):
	def __init__(self, data_root, trials, windows, types=None):
		super(GazeDataset, self).__init__()
		self.data_root = data_root
		self.trials = trials
		self.windows = windows
		self.types = types

		# Pre compute the length during initialization
		self._len = 0
		self._trial_lengths = []
		for trial in self.trials:
			l = len(os.listdir(trial))
			self._trial_lengths.append((trial, l))
			self._len += l - np.max(self.windows)
		self.inds = range(0, self._len)

	def __getitem__(self, index):
		index = self.inds[index]
		for t in self._trial_lengths:
			trial = t[0]
			trial_length = t[1]
			index += np.max(self.windows)
			tmp_ind = index - trial_length
			if tmp_ind < 0:
				name =  '{:05d}'.format(tmp_ind) + '.png'
				egoframe = io.imread(os.path.join(self.data_root, 'gazemap', '_'.join(trial.split('/')), name))
				return egoframe
			index = tmp_ind

	def __len__(self):
		return self._len

	def set_inds(self, inds):
		self.inds = inds
		self._len = len(inds)
