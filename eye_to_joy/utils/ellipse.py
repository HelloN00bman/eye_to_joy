import matplotlib.pyplot as plt

import numpy as np

import scipy.ndimage as imf

import skimage.draw as draw


class GazeMapper(object):
	def __init__(self, w=224, h=224, ax_x=4, ax_y=6, sigma=3):
		self.w, self.h = [w, h]
		self.ax_y, self.ax_x = [ax_x, ax_y]
		self.sigma = sigma		
		self.mask = np.zeros((self.w, self.h))

	def _create_mask(self, x, y):
		mask = np.zeros((self.w, self.h))
		ell = draw.ellipse(x, y, self.ax_x, self.ax_y)
		mask[ell] = 1
		mask = imf.gaussian_filter(mask, self.sigma)
		return mask

	def zero_and_one(self, arr):
		return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

	def create_masks(self, pts):
		for pt in pts:
			x_ind, y_ind = pt
			m = self._create_mask(x_ind, y_ind)
			self.mask = self.mask + m
		self.mask = self.zero_and_one(self.mask)
		return self.mask

def main():
	points = np.array([[100,100], [50,50], [103,102
	mask = create_masks(points)

if __name__ == '__main__':
	main()