import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import scipy.ndimage as imf

import skimage.draw as draw
import scipy.stats as stats


import cv2

class GazeMapper(object):
	def __init__(self, w=224, h=224, ax_x=4, ax_y=6, sigma=3):
		self.w, self.h = [w, h]
		self.ax_y, self.ax_x = [ax_x, ax_y]
		self.sigma = sigma		
		self.mask = np.zeros((self.w, self.h))

	def _create_mask(self, x, y):
		mask = np.zeros((self.w, self.h))
		mask = cv2.ellipse(mask, (x,224-y), (self.ax_x, self.ax_y), 0, 0, 360, 255 , -1)
		mask = imf.gaussian_filter(mask, self.sigma)
		return mask

	def zero_and_one(self, arr):
		denom = 1 if  (np.max(arr) - np.min(arr)) == 0 else (np.max(arr) - np.min(arr))
		return (arr - np.min(arr)) / denom

	def create_masks(self, pts):
		for pt in pts:
			if pt[0] >= 0 and pt[1] >= 0:
				x_ind, y_ind = pt
				m = self._create_mask(x_ind, y_ind)
				self.mask = self.mask + m
		self.mask = self.zero_and_one(self.mask)
		return self.mask

def main():
	points = np.array(([ 93, 116],
	 [ 93, 118],
	 [ 93, 120],
	 [ 93, 121],
	 [ 93, 123],
	 [ 93, 124],
	 [ 93, 126],
	 [ -1,  -1],
	 [ 92, 127],
	 [ -1,  -1],
	 [ 93, 128],
	 [ 92, 130],
	 [ -1,  -1],
	 [ -1,  -1],
	 [ 91, 133],
	 [ 92, 132],
	 [ 92, 133],
	 [ -1,  -1],
	 [ -1,  -1],
	 [ 95, 129],
	 [ 92, 131],
	 [ -1,  -1],
	 [ 93, 131],
	 [ 93, 131],
	 [ 93, 130],
	 [ 94, 129],
	 [ 94, 128],
	 [ 94, 127],
	 [ 94, 127],
	 [ -1,  -1],
	 [ 95, 115],
	 [ 95, 115],
	 [ 95, 114],
	 [ -1,  -1],
	 [ 95, 114],
	 [ 96, 114]))
	#points = np.array(([105, 105, 105, 105, 104, 104, 104, 104, 103], [72, 72, 73, 73, 72, 72, 72, 72, 67])).T
	#points = np.array([[100,100], [50,50], [103,102]])
	gm = GazeMapper()
	mask = gm.create_masks(points)
	plt.imshow(mask)
	plt.savefig('img/test.png')
	plt.close()

if __name__ == '__main__':
	main()
