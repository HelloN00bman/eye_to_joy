import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.models as models
import functools as ft

def weight_init(layer, genre, init_func):
	if isinstance(layer, genre):
		init_func(layer.weight)

def bias_init(layer, genre, init_func):
	if isinstance(layer, genre):
		init_func(layer.bias)

class NaiveMapper(nn.Module):
	def __init__(self, input_size, num_classes=257, batch_size=10, hidden_size=512, future_length=1):
		super(NaiveMapper, self).__init__()
		
		self.input_size = input_size
		self.future_length = future_length

		self.num_classes = num_classes
		self.hidden_dim = hidden_size
		self.minibatch = batch_size
		
		self.hidden = self.init_hidden()
		
		self.features = self._make_features()
		self.classifier = self._make_classifier()
		self.pos_top = self._make_pos_classifier()
		self.mode_top = self._make_mode_classifier()
		
		self._init_weights_and_biases()

	def init_hidden(self):
		# CHANGE THE 10 TO A VARIABLE DUMMY
		hidden = (
			autograd.Variable(torch.zeros(2, 10, self.hidden_dim).float()),
			autograd.Variable(torch.zeros(2, 10, self.hidden_dim).float())
		)
		
		hidden = [l.cuda() if torch.cuda.is_available() else l for l in hidden]

		return hidden

	def _init_weights_and_biases(self):
		init_w = nn.init.xavier_normal_
		init_b = ft.partial(nn.init.constant_, val=0)
		
		for param in self.features.parameters():
			if len(param.shape) >= 2:
				init_w(param.data)
			else:
				init_b(param.data)

		self.classifier.apply(ft.partial(weight_init, genre=nn.Linear, init_func=init_w))
		self.classifier.apply(ft.partial(bias_init, genre=nn.Linear, init_func=init_b))

		self.pos_top.apply(ft.partial(weight_init, genre=nn.Linear, init_func=init_w))
		self.pos_top.apply(ft.partial(bias_init, genre=nn.Linear, init_func=init_b))

		self.mode_top.apply(ft.partial(weight_init, genre=nn.Linear, init_func=init_w))
		self.mode_top.apply(ft.partial(bias_init, genre=nn.Linear, init_func=init_b))


	def _make_features(self):
		lstm = nn.LSTM(input_size=512*14*14 + 1024*3, hidden_size=self.hidden_dim, num_layers=2)
		return lstm


	def _make_classifier(self):
		layers = nn.Sequential(
			nn.Linear(self.hidden_dim, 1024),
			nn.ReLU(True),
			nn.BatchNorm1d(1024),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024,self.num_classes+1*10)
		)
		return layers

	def _make_pos_classifier(self):
		layers = nn.Sequential(
			nn.Linear(self.hidden_dim, 1024),
			nn.ReLU(True),
			# nn.Dropout(p=0.5),
			nn.BatchNorm1d(1024),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.BatchNorm1d(1024),
			nn.Linear(1024,(self.num_classes)*10)
			)
		return layers

	def _make_mode_classifier(self):
		layers = nn.Sequential(
			nn.Linear(self.hidden_dim, 1024),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024,3*10)
			)
		return layers	

	# def forward(self, x):
	# 	lstm_out, hidden = self.features(x, self.hidden)
	# 	print(lstm_out.shape)
	# 	feat_out = self.classifier(lstm_out.view(-1, lstm_out.size(2))).view(1000,25)
	# 	pos = feat_out[:,:self.num_classes-1]
	# 	mode = feat_out[:,self.num_classes-1]
	# 	return pos, mode, hidden

	def forward(self, x, hidden=None):
		if not hidden: 
			hidden = self.hidden
		lstm_out, hidden = self.features(x, hidden)
		lstm_out = lstm_out.view(-1, self.hidden_dim)

		pos = self.pos_top(lstm_out).view(-1, self.num_classes)
		mode = self.mode_top(lstm_out).view(-1, 3)
		return pos, mode, hidden