
import torch
import torch.nn as nn
import functools as ft


def weight_init(layer, genre, init_func):
	if isinstance(layer, genre):
		init_func(layer.weight)

def bias_init(layer, genre, init_func):  
	if isinstance(layer, genre):
		init_func(layer.bias)



class GazeEncoder(nn.Module):
	def __init__(self, w):
		super(GazeEncoder, self).__init__()
		self.w = w
		self.features = self._make_features()
		self.classifier = self._make_classifier()
		self._init_weights_and_biases()

	def _init_weights_and_biases(self):
		init_w = nn.init.xavier_normal_
		init_b = ft.partial(nn.init.constant_, val=0)
	
		self.features.apply(ft.partial(weight_init, genre=nn.Conv1d, init_func=init_w))
		self.features.apply(ft.partial(bias_init, genre=nn.Conv1d, init_func=init_b))
	
		self.classifier.apply(ft.partial(weight_init, genre=nn.Linear, init_func=init_w))
		self.classifier.apply(ft.partial(bias_init, genre=nn.Linear, init_func=init_b))
		
	def _make_features(self):
		in_channels = 2
		layers = nn.Sequential(
			nn.Conv1d(in_channels, 16, kernel_size=3),
			nn.BatchNorm1d(16),
			nn.ReLU(inplace=True),
			nn.Conv1d(16, 16, kernel_size=3),
			nn.BatchNorm1d(16),
			nn.ReLU(inplace=True),
			nn.Conv1d(16, 16, kernel_size=1, padding=1),
			nn.BatchNorm1d(16),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),
			nn.Conv1d(16, 32, kernel_size=3),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True),
			nn.Conv1d(32, 32, kernel_size=3),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True),
			nn.Conv1d(32, 32, kernel_size=1, padding=1),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),	
			)
		return layers

	def _make_classifier(self):
		num_classes = 2
		layers = nn.Sequential(
			# nn.Linear(int(64*self.w/4), 1024),
			nn.Linear(64, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			# nn.Dropout(p=0.5),
			nn.Linear(1024, 1024),
			)
		return layers

	def forward(self, x):
		print(x.shape)
		x = self.features(x).view(10, -1)
		print(x.shape)
		# x = x.view(-1, int(64*self.w/4))
		# x = x.view(-1)
		
		# x = self.classifier(x)
		return x

