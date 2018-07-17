import numpy as np
import os
import os.path
import torch
from torch.utils.data import *
from eye_to_joy.utils import *

class JoyDataset(Dataset):
	def __init__(self, trials, windows, types=None):
		super(GazeDataset, self).__init__()
		self.trials = trials
		self.windows = windows
		self.types = types

		# Pre compute the length during initialization
		self._len = 0
		self._trial_lengths = []
		for trial in self.trials:
			l = torch.load(trial + '/00001.pt7').shape[0]
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
				name = '_'.join(trial.split('/')) + '.pt3'
				max_tensor = torch.Tensor(4, np.max(self.windows)+1)
				tmp = torch.load(trial + name)
				max_tensor = torch.t(tmp[index-np.max(self.windows):index+1, :])
				return max_tensor
			index = tmp_ind

	def __len__(self):
		return self._len

	def set_inds(self, inds):
		self.inds = inds
		self._len = len(inds)
