
import torch
import torch.nn as nn

from torch.autograd import Variable
from .EgoEncoder import *
from .GazeEncoderMS import *

WINDOWS = [12, 24, 36]
WINDOW_SIZE = max(WINDOWS)

class SpatSemLSTM(nn.Module):
	def __init__(self):
		print('SpatSem init')
		super(SpatSemLSTM, self).__init__()
		self.hidden_dim = 256
		print('make features')
		self._make_features()
		print('make classifier')
		self._make_classifier()
		pass

	def init_hidden(self, dim):
		return (Variable(torch.zeros(1, 1, dim).float().cuda()),
				Variable(torch.zeros(1, 1, dim).float().cuda()))

	def _make_features(self):
		self.ego = EgoEncoder('vgg13', True)
		#self.ego.requires_grad = False
		self.sem_lstm = nn.LSTM(4096, self.hidden_dim)
		self.gaze = GazeEncoderMS(*WINDOWS)
		#self.gaze.requires_grad = False
		self.spat_lstm = nn.LSTM(self.hidden_dim + 1024*3, self.hidden_dim)

	def _make_classifier(self):
		num_classes = 2
		layers = nn.Sequential(
			nn.Linear(256, 512),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(512, 2),
			nn.Tanh(),
			)
		self.classifier = layers
		return layers

	def forward(self, count, ego, gaze1, gaze2, gaze3, sem_out, sem_hidden, spat_hidden):
		if not count % WINDOW_SIZE:
			ego = ego.view(-1,4,224,224)
			ego_out = self.ego(ego)
			ego_out = ego_out.view(1,1,-1)
			sem_out, sem_hidden = self.sem_lstm(ego_out, sem_hidden)

		gaze_out = self.gaze(gaze1, gaze2, gaze3)
		gaze_out = gaze_out.view(1,1,-1)
		ego_gaze = torch.cat((sem_out, gaze_out), 2)

		spat_out, spat_hidden = self.spat_lstm(ego_gaze, spat_hidden)
		spat_out = spat_out.view(1, -1)
		out = self.classifier(spat_out)
		return out, sem_out, sem_hidden, spat_hidden

