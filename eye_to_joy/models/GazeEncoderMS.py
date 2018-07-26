from .GazeEncoder import GazeEncoder

import torch
import torch.nn as nn


class GazeEncoderMS(nn.Module):
	def __init__(self, w1, w2, w3, eval=False):
		super(GazeEncoderMS, self).__init__()
		self.w1 = w1
		self.w2 = w2
		self.w3 = w2		
		self.ge1 = GazeEncoder(w1)
		self.ge2 = GazeEncoder(w2)
		self.ge3 = GazeEncoder(w3)
		if eval:
			self.ge1 = self.ge1.eval()
			self.ge2 = self.ge2.eval()
			self.ge3 = self.ge3.eval()
		self.classifier = self._make_classifier()
		self._init_weights()
		pass

	def _init_weights(self):
		cls_keys = self.classifier.state_dict().keys()
		tmp_cls_state_dict = dict()
		for key in cls_keys:
			vals = self.classifier.state_dict()[key]
			if 'weight' in key:
				nn.init.xavier_normal_(vals)
			else:
				nn.init.constant_(vals, 0)
			tmp_cls_state_dict[key] = vals
		self.classifier.load_state_dict(tmp_cls_state_dict)

	def _make_classifier(self):
		num_classes = 2
		layers = nn.Sequential(
			nn.Linear(1024*3, 64),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(64, 64),
			nn.ReLU(True),
            nn.Dropout(),
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

	def classify(self, x1, x2, x3):
		x = self.featurize(x1, x2, x3)
		x = self.classifier(x)
		return x

