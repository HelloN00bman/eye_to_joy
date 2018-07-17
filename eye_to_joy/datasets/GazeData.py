import csv

import numpy as np

import os
import os.path

import sys

class GazeData(object):
	def __init__(self, inds=[0,4,5], round=[0,1,1], w=224, h=224):
		self.path = ''
		self.header = np.array([])
		self.data = np.array([])
		self.inds = inds
		self.round = round
		self.w = w
		self.h = h

	def _clean_data(self):
		x = self.data[:,1]
		too_big_x = np.where(x >= self.w)
		too_small_x = np.where(x < 0)

		y = self.data[:,2]
		too_big_y = np.where(y >= self.h)
		too_small_y = np.where(y < 0)

		x[too_big_x] = -1
		x[too_big_y] = -1
		x[too_small_x] = -1
		x[too_small_y] = -1

		y[too_big_x] = -1
		y[too_big_y] = -1
		y[too_small_x] = -1
		y[too_small_y] = -1
		
		self.data[:,1] = x
		self.data[:,2] = y

	def _get_column(self, ind, data, r):
		print(np.array(data).shape)
		data = np.array(data)
		if len(data.shape) > 1 and r:
			return list(np.round(data[:,ind].astype('float64')*224))
		elif len(data.shape) > 1:
			return list(data[:,ind].astype('float64'))
		else:
			## Should just be the header
			return data[ind]

	def _get_columns(self, data):
		tmp_data = []
		for ind, r in zip(self.inds, self.round):
			tmp = self._get_column(ind, data, r)
			tmp_data += [tmp]
		return np.array(tmp_data).transpose()

	def load_csv(self, path):
		self.path = path
		data = []
		with open(self.path, 'rb') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]
		if np.array(data[1:]).shape[0] == 0:
			return
		self.header = self._get_columns(data[0])
		self.data = self._get_columns(data[1:])
		self._clean_data()


##################################################################################
##  DESCRIBE MAIN FUNCTION HERE
##  1) Use a test path
##  2) Create a GazeData object with the default inputs
##  3) Load the data at data_path
##  4) Assert that header and data are correct sizes
##################################################################################

def main():
	data_path = '/mnt/sdb1/bnewman1/harpdata/cleaned_data/p01/002/traj_info.csv'
	gd = GazeData(inds = [0,2,3], round=[0,0,0], w=2, h=2)
	gd.load_csv(data_path)
	#assert(gd.header.shape[0] == 3 and gd.header.size == gd.header.shape[0])
	#assert(gd.data.shape[0] == 3 and gd.data.shape[1] == 3793)
	import IPython as ipython; ipython.embed()

if __name__ == '__main__':
	main()