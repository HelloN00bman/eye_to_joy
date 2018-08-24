
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.models as models

from .EgoEncoder import *
from .GazeEncoderMS import *

WINDOWS = [12, 24, 36]
WINDOW_SIZE = max(WINDOWS)

class NaiveMapper(nn.Module):
	def __init__(self):
		super(NaiveMapper, self).__init__()
		self.hidden_dim = 512
		self.minibatch = 10
		self._make_features()
		self._make_classifier()

	def init_hidden(self):
		if not torch.cuda.is_available():
			return (autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float()),
				autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float()))
		else:
			return (autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float().cuda()),
				autograd.Variable(torch.zeros(2, self.minibatch, self.hidden_dim).float().cuda()))

	def _init_weights_and_biases(self):
		init_w = nn.init.xavier_normal_
		init_b = ft.partial(nn.init.constant_, val=0)
		
		for param in self.lstm.parameters():
			if len(param.shape) >= 2:
				init_w(param.data)
			else:
				init_b(param.data, 0)

		self.classifier.apply(ft.partial(weight_init, nn.Linear, init_w))
		self.classifier.apply(ft.partial(bias_init, nn.Linear, init_b))

	def _make_features(self):
		self.ego = models.vgg13_bn(True) # pretained VGG model w/ batchnorm
		self.ego.eval()

		self.gaze = GazeEncoderMS(*WINDOWS)

		self.lstm = nn.LSTM(input_size=512*7*7 + 1024*3, hidden_size=self.hidden_dim, num_layers=2)

	def _make_classifier(self):
		num_classes = 257
		layers = nn.Sequential(
			nn.Linear(self.hidden_dim, 1024),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024,num_classes)
			)
		self.classifier = layers
		return layers

	def forward(self, x, hidden):
		lstm_out, lstm_hidden = self.lstm(x, hidden)
		out = self.classifier(lstm_out)
		# out = F.log_softmax(out)
		out1 = out[:,:,:257]
		# out2 = nn.Sequential(nn.Sigmoid())(out[:,:,2])
		out2 = out[:,:,257]
		return out1, out2, lstm_hidden

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
