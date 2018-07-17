
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.models as models

from .EgoEncoder import *
from .GazeEncoderMS import *

WINDOWS = [12, 24, 36]
WINDOW_SIZE = max(WINDOWS)

class NaiveMapper(nn.Module):
	def __init__(self):
		super(NaiveMapper, self).__init__()
		self.hidden_dim = 256
		self.minibatch = 1
		self.hidden = self.init_hidden()
		self._make_features()
		self._make_classifier()
		pass

	def init_hidden(self):
		return (autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float().cuda()),
				autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float().cuda()))

	def _make_features(self):
		# self.ego = EgoEncoder('vgg13', True)
		self.ego = models.vgg13(True) # pretain the model
		self.gaze = GazeEncoderMS(*WINDOWS)
		self.lstm = nn.LSTM(input_size=512*512 + 1024*3, hidden_size=self.hidden_dim, num_layers=2)

	def _make_classifier(self):
		num_classes = 2
		layers = nn.Sequential(
			nn.Linear(self.hidden_dim, 256),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(256, 2),
			)
		self.classifier = layers
		return layers

	def forward(self, ego, gaze1, gaze2, gaze3):
		ego = ego.view(1,3,224,224)
		ego_out = self.ego(ego)
		ego_out = ego_out.view(1,1,-1)
		gaze_out = self.gaze(gaze1, gaze2, gaze3)
		gaze_out = gaze_out.view(1,1,-1)
		ego_gaze = torch.cat((ego_out, gaze_out),2)
		lstm_out, self.lstm_hidden = self.lstm(ego_gaze, self.hidden)
		lstm_out = lstm_out.view(1, -1)
		out = self.classifier(lstm_out)
		return out
