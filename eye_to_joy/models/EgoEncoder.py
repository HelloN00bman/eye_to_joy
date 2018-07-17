import torch
import torch.nn as nn
import torchvision.models as models


class EgoEncoder(nn.Module):
	def __init__(self, base_model, pretrained=True):
		super(EgoEncoder, self).__init__()
		self.pretrain = pretrained
		try:
			if base_model == 'vgg11':
				self.vgg11()
			elif base_model == 'vgg13':
				self.vgg13()
			elif base_model == 'vgg16':
				self.vgg16()
			elif base_model == 'vgg19':
				self.vgg19()
		except ValueError as e:
			print('Please provide a proper model type.')
			print(e)

	def _transfer_features(self, layers, state_dict):
		# Transfer the features from pretrained model to EgoEncoder
		tmp_state_dict = layers.state_dict()
		keys = state_dict.keys()
		for key in keys:
			# Transfer over all of the weights from VGG and add 
			# a channel of Xavier initialized weights
			if key == '0.weight': # could make this more general
				cur_feats = state_dict[key]
				cur_feats_shape = cur_feats.shape
				tmp_feats_shape = [cur_feats_shape[0], 
									1, # could make this more general
									cur_feats_shape[2], 
									cur_feats_shape[3]]
				tmp_feats = torch.Tensor(*tmp_feats_shape)
				nn.init.xavier_normal(tmp_feats)
				tmp_state_dict[key] = torch.cat((cur_feats, tmp_feats),1)
			else:
				tmp_state_dict[key] = state_dict[key]
		layers.load_state_dict(tmp_state_dict)
		return layers

	def _transfer_classifier(self, layers, state_dict):
		# Transfer the classifier weights from the pretrained classifier
		tmp_state_dict = layers.state_dict()
		keys = state_dict.keys()
		for key in keys:
			tmp_state_dict[key] = state_dict[key]
		return layers

	def _set_features_and_classifier(self, model, features=None, classifier=None):
		# Transfer weights from the pretrained model
		tmp_feat_state_dict = model.features.state_dict()
		tmp_cls_state_dict = model.classifier.state_dict()
		self.features = self._transfer_features(features, tmp_feat_state_dict)
		self.classifier = self._transfer_classifier(classifier, tmp_cls_state_dict)

	def _make_features(self): # Could make this more general
		in_channels = 4
		layers = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),	
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),		
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
			)
		return layers

	def _make_classifier(self):
		layers = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096)
			)
		return layers

	def vgg11(self):
		print('Downloading vgg11')
		model = models.vgg11(True)
		pass

	def vgg13(self):
		print('Downloading vgg13')
		model = models.vgg13(self.pretrain)
		self._set_features_and_classifier(model, self._make_features(), self._make_classifier())
		pass

	def vgg16(self):
		print('Downloading vgg16')
		model = models.vgg16(True)
		pass

	def vgg19(self):
		print('Downloading vgg19')
		model = models.vgg19(True)
		pass

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x




