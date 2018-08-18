
import torch
import torch.nn as nn
import functools as ft


def weight_init(layer, genre, init_func):
	print(layer, genre, init_func)
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
		
		# feat_keys = self.features.state_dict().keys()
		# tmp_feat_state_dict = dict()
		# for key in feat_keys:
		# 	vals = self.features.state_dict()[key]
		# 	if 'weight' in key:
		# 		nn.init.xavier_normal_(vals)
		# 	else:
		# 		nn.init.constant_(vals, 0)
		# 	tmp_feat_state_dict[key] = vals
		# self.features.load_state_dict(tmp_feat_state_dict)

		# cls_keys = self.classifier.state_dict().keys()
		# tmp_cls_state_dict = dict()
		# for key in cls_keys:
		# 	vals = self.classifier.state_dict()[key]
		# 	if 'weight' in key:
		# 		nn.init.xavier_normal_(vals)
		# 	else:
		# 		nn.init.constant_(vals, 0)
		# 	tmp_cls_state_dict[key] = vals
		# self.classifier.load_state_dict(tmp_cls_state_dict)

	def _make_features(self):
		in_channels = 2
		layers = nn.Sequential(
			nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True),
			nn.Conv1d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),
			# nn.Conv1d(32, 32, kernel_size=1, padding=0),
			nn.Conv1d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.Conv1d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),	
			# nn.Conv1d(64, 64, kernel_size=1, padding=0),
			)
		return layers

	def _make_classifier(self):
		num_classes = 2
		layers = nn.Sequential(
			nn.Linear(64*self.w/4, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			# nn.Dropout(p=0.5),
			nn.Linear(1024, 1024),
			)
		return layers

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 64*self.w/4)
		print(x.shape)
		
		x = self.classifier(x)
		return x

