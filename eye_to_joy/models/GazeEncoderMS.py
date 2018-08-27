from .GazeEncoder import GazeEncoder

import torch
import torch.nn as nn

import functools as ft

def weight_init(layer, genre, init_func):
	if isinstance(layer, genre):
		init_func(layer.weight)

def bias_init(layer, genre, init_func):
	if isinstance(layer, genre):
		init_func(layer.bias)

class GazeEncoderMS(nn.Module):
	def __init__(self, w1, w2, w3):
		super(GazeEncoderMS, self).__init__()
		self.w1 = w1
		self.w2 = w2
		self.w3 = w2		

		self.ge1 = GazeEncoder(w1)
		self.ge2 = GazeEncoder(w2)
		self.ge3 = GazeEncoder(w3)

		self.classifier = self._make_classifier()
		self._init_weights_and_biases()

	def _init_weights_and_biases(self):
		init_w = nn.init.xavier_normal_
		init_b = ft.partial(nn.init.constant_, val=0)

		self.classifier.apply(ft.partial(weight_init, genre=nn.Linear, init_func=init_w))
		self.classifier.apply(ft.partial(bias_init, genre=nn.Linear, init_func=init_b))

	def _make_classifier(self):
		num_classes = 2
		layers = nn.Sequential(
			nn.Linear(1024*3, 64),
			nn.BatchNorm1d(32),
			nn.ReLU(True),
			nn.Linear(64, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(True),
			nn.Linear(64, num_classes),
			# nn.Tanh(),
			)
		return layers

	def featurize(self, x1, x2, x3):
		x1 = self.ge1(x1)
		x2 = self.ge2(x2)
		x3 = self.ge3(x3)
		x = torch.cat((x1,x2,x3), 1)
		return x

	def classify(self, features):
		x = self.classifier(features)
		return x

	def forward(self, x1, x2, x3):
		x = self.featurize(x1, x2, x3)
		x = self.classifier(x)
		return x

