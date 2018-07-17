
import torch
import torch.nn as nn


class GazeEncoder(nn.Module):
	def __init__(self, w):
		super(GazeEncoder, self).__init__()
		self.w = w
		self.features = self._make_features()
		self.classifier = self._make_classifier()
		self._init_network()
		pass

	def _init_network(self):
		feat_keys = self.features.state_dict().keys()
		tmp_feat_state_dict = dict()
		for key in feat_keys:
			vals = self.features.state_dict()[key]
			if 'weight' in key:
				nn.init.xavier_normal(vals)
			else:
				nn.init.constant(vals, 0)
			tmp_feat_state_dict[key] = vals
		self.features.load_state_dict(tmp_feat_state_dict)

		cls_keys = self.classifier.state_dict().keys()
		tmp_cls_state_dict = dict()
		for key in cls_keys:
			vals = self.classifier.state_dict()[key]
			if 'weight' in key:
				nn.init.xavier_normal(vals)
			else:
				nn.init.constant(vals, 0)
			tmp_cls_state_dict[key] = vals
		self.classifier.load_state_dict(tmp_cls_state_dict)

	def _make_features(self):
		in_channels = 2
		layers = nn.Sequential(
			nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv1d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),
			# nn.Conv1d(32, 32, kernel_size=1, padding=0),
			nn.Conv1d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv1d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),	
			# nn.Conv1d(64, 64, kernel_size=1, padding=0),
			)
		return layers

	def _make_classifier(self):
		num_classes = 2
		layers = nn.Sequential(
			nn.Linear(64 * self.w / 4, 1024),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(1024, 1024),
			)
		return layers

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 64*self.w/4)
		x = self.classifier(x)
		return x

