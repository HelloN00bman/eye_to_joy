
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
		self.minibatch = 10
		# self.hidden = self.init_hidden()
		self._make_features()
		self._make_classifier()
		pass

	def init_hidden(self, cuda):
		if not cuda:
			return (autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float()),
				autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float()))
		else:
			return (autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float().cuda()),
				autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float().cuda()))

	def _make_features(self):
		# self.ego = EgoEncoder('vgg13', True)
		self.ego = models.vgg13(True) # pretain the model
		self.gaze = GazeEncoderMS(*WINDOWS)
		self.lstm = nn.LSTM(input_size=512*7*7 + 1024*3, hidden_size=self.hidden_dim, num_layers=2)

	def _make_classifier(self):
		num_classes = 3
		layers = nn.Sequential(
			nn.Linear(self.hidden_dim, 256),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(256, num_classes),
			)
		self.classifier = layers
		return layers

	def forward(self, x, hidden):
		lstm_out, lstm_hidden = self.lstm(x, hidden)
		out = self.classifier(lstm_out)
        out = nn.functional.log_softmax(out)
		return out, lstm_hidden

		# ego_out = torch.unbind(ego.view(1, 1, -1))[0]
		# print('shapes')
		# print(ego_out.shape)
		# print(gaze.shape)
		# ego_gaze = torch.unsqueeze(torch.cat((ego_out, gaze),1),0)
		# print(ego_gaze.shape)
		# lstm_out, self.lstm_hidden = self.lstm(ego_gaze, self.hidden)
		# lstm_out = lstm_out.view(1, -1)
		# out = self.classifier(lstm_out)
		# return out
