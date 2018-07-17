import torch
import torch.autograd as autograd
import torch.nn as nn
import torchvision.models as models
import numpy as np

import IPython

class Featurizer(nn.Module):
	def __init__(self, model_type):
		super(Featurizer, self).__init__()

		try:
			if model_type == 'vgg11':
				self.vgg11()
			elif model_type == 'vgg13':
				self.vgg13()
			elif model_type == 'vgg16':
				self.vgg16()
			elif model_type == 'vgg19':
				self.vgg19()
		except:
			print('Please provide a proper model type.')
		pass

	def _set_features_and_classifier(self, model, features=None, classifier=None):
		self.features = features
		self.classifier = classifier

	def vgg11(self):
		print('Downloading vgg11')
		model = models.vgg11(True)
		self._set_features_and_classifier(model, model.features, model.classifier[0])

	def vgg13(self):
		print('Downloading vgg13')
		model = models.vgg13(True)
		self._set_features_and_classifier(model, model.features, model.classifier[0])

	def vgg16(self):
		print('Downloading vgg16')
		model = models.vgg16(True)
		self._set_features_and_classifier(model, model.features, model.classifier[0])
		
	def vgg19(self):
		print('Downloading vgg19')
		model = models.vgg19(True)
		self._set_features_and_classifier(model, model.features, model.classifier[0])

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

def main():
	vgg13 = Featurizer('vgg13')
	inp = torch.Tensor(np.random.randn(1,3,224,224)).cpu()
	out = vgg13.forward(autograd.Variable(inp, volatile=True)).data
	print(inp.shape)
	print(out.shape)
	IPython.embed()

if __name__ == '__main__':
	main()


