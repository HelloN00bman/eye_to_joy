import torch
import numpy as np

from torch.utils.data import *
from torch.utils.trainer import *
from torch.autograd import Variable

class SpatSemTrainer(Trainer):
	def __init__(self,  model=None, criterion=None, optimizer=None, dataset=None):
		super(SpatSemTrainer, self).__init__(model, criterion, optimizer, dataset)

	def train(self):
		for i, data in enumerate(self.dataset, self.iterations + 1):
			gaze, ego, labels = data
			gaze = np.array(gaze)
			self.call_plugins('batch', i, gaze, ego, labels)
			for j in range(gaze.shape[0]):

				gaze1_var = Variable(gaze[j,24:, :])
				gaze2_var = Variable(gaze[j,12:, :])
				gaze3_var = Variable(gaze[j, :, :])
				ego_var = Variable(ego[j/36])
				labels_var = Variable(labels[j])

				plugin_data = [None, None]

				def closure():
					batch_output = self.model(input_var)
					loss = self.criterion(batch_output, target_var)
					loss.backward()
					if plugin_data[0] is None:
						plugin_data[0] = batch_output.data
						plugin_data[1] = loss.data
					return loss

				self.optimizer.zero_grad()
				self.optimizer.step(closure)
				self.call_plugins('iteration', i, batch_input, batch_target,
									*plugin_data)
				self.call_plugins('update', i, self.model)

		self.iterations += i
		pass