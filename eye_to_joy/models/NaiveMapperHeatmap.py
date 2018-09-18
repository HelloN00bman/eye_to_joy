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

class NaiveMapperHeatmap(nn.Module):
	def __init__(self, num_classes=257, batch_size=10, hidden_size=512, future_length=1):
		super(NaiveMapperHeatmap, self).__init__()

		self.ngf = 2
		self.sf = 2
		self.nc = 10

		self.num_classes = num_classes
		self.hidden_dim = hidden_size
		self.minibatch = batch_size
		self.future_length = future_length
		
		self._make_pos_classifier()

		self.hidden = self.init_hidden()
		
		self.features = self._make_features()
		self.decode1, self.decode2 = self.decoder()
		self.mode_top = self._make_mode_classifier()

		
		# self._init_weights_and_biases()

	def init_hidden(self):
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

		self.decode1.apply(ft.partial(weight_init, genre=nn.Linear, init_func=init_w))
		self.decode1.apply(ft.partial(bias_init, genre=nn.Linear, init_func=init_b))

		self.decode2.apply(ft.partial(weight_init, genre=nn.Conv2d, init_func=init_w))
		self.decode2.apply(ft.partial(bias_init, genre=nn.Conv2d, init_func=init_b))

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
		self.d1 = nn.Linear(self.hidden_dim, self.ngf*8*2*4*4)

		self.up1 = nn.UpsamplingNearest2d(scale_factor=self.sf)
		self.pd1 = nn.ReplicationPad2d(1)
		self.d2 = nn.Conv2d(self.ngf*8*2, self.ngf*8, 3, 1)
		self.bn6 = nn.BatchNorm2d(self.ngf*8, 1.e-3)

		self.up2 = nn.UpsamplingNearest2d(scale_factor=self.sf)
		self.pd2 = nn.ReplicationPad2d(1)
		self.d3 = nn.Conv2d(self.ngf*8, self.ngf*4, 3, 1)
		self.bn7 = nn.BatchNorm2d(self.ngf*4, 1.e-3)

		self.up3 = nn.UpsamplingNearest2d(scale_factor=self.sf)
		self.pd3 = nn.ReplicationPad2d(1)
		self.d4 = nn.Conv2d(self.ngf*4, self.ngf*2, 3, 1)
		self.bn8 = nn.BatchNorm2d(self.ngf*2, 1.e-3)

		self.up4 = nn.UpsamplingNearest2d(scale_factor=self.sf)
		self.pd4 = nn.ReplicationPad2d(1)
		self.d5 = nn.Conv2d(self.ngf*2, self.ngf, 3, 1)
		self.bn9 = nn.BatchNorm2d(self.ngf, 1.e-3)

		self.up5 = nn.UpsamplingNearest2d(scale_factor=self.sf)
		self.pd5 = nn.ReplicationPad2d(1)
		self.d6 = nn.Conv2d(self.ngf, self.nc, 3, 1)

		self.leakyrelu = nn.LeakyReLU(0.2)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def _make_mode_classifier(self):
		print(self.future_length)
		layers = nn.Sequential(
			nn.Linear(self.hidden_dim, 1024),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(1024, 1024),
			nn.ReLU(True),
			nn.Linear(1024,1*self.future_length)
			)
		return layers	

	def decoder(self):
		decode1 = nn.Sequential(
			self.d1,
			self.relu,
		)	

		decode2 = nn.Sequential(
			self.up1,
			self.pd1,
			self.d2,
			self.bn6,
			self.leakyrelu,
			self.up2,
			self.pd2,
			self.d3,
			self.bn7,
			self.leakyrelu,
			self.up3,
			self.pd3,
			self.d4,
			self.bn8,
			self.leakyrelu,
			self.up4,
			self.pd4,
			self.d5,
			self.bn9,
			self.leakyrelu,
			self.up5,
			self.pd5,
			self.d6,
			self.sigmoid
		)

		return decode1, decode2

	def forward(self, x, hidden=None):
		if not hidden: 
			hidden = self.hidden
		lstm_out, hidden = self.features(x, hidden)
		lstm_out = lstm_out.view(-1, self.hidden_dim)
		decode1_out = self.decode1(lstm_out)
		decode1_out = decode1_out.view(-1, self.ngf*8*2, 4, 4)
		pos = self.decode2(decode1_out)
		mode = self.mode_top(lstm_out.view(-1, self.hidden_dim)).view(-1)
		return pos, mode, hidden