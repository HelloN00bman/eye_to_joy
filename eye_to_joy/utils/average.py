import numpy as np
import matplotlib.pyplot as plt


class Average(object):
	def __init__(self, data):
		self.data = data
		self.window_size = np.shape(self.data)[0]
		self.avg_type = ''
		self.domain = np.linspace(0,0,0)

	def _Gaussian(self, mu, sigma, **kwargs):
		self._weights = (1./np.sqrt(2. * np.pi * np.power(sigma, 2.))) * np.exp(-np.power(np.linspace(0.,self.window_size, self.window_size) - mu, 2.) / (2. * np.power(sigma,2.))) 
		self._weights = self._weights / sum(self._weights)

	def _recency(self, exp, **kwargs):
		self._weights = np.power(self.domain, exp) / np.sum(np.power(self.domain, exp))

	def _primacy(self, exp, **kwargs):
		if exp == 0:
			self._recency(exp)
		else:
			self._weights = (np.power(self.domain[-1]-self.domain, exp)) / np.sum(np.power(self.domain[-1]-self.domain, exp))

	def _uniform(self, **kwargs):
		self._weights = np.ones(len(self.data)) / (self.window_size * 1.0)

	def avg(self, avg_type, **kwargs):
		self.avg_type = avg_type
		self.domain = np.linspace(0, len(self.data), self.window_size)
		
		if avg_type == 'Uniform':
			self._uniform(**kwargs)
		elif avg_type == 'Primacy':
			self._primacy(**kwargs)
		elif avg_type == 'Recency':
			self._recency(**kwargs)	
		elif avg_type == 'Gaussian':
			self._Gaussian(**kwargs)

		self.average = self._weights * self.data
		return self.average

def main():
	x = np.random.randn(10)
	avg = Average(x)
	print('data:', x)
	print('sum:', sum(x))
	print('uniform:', avg.avg('Uniform'))
	mu = len(x)/2.
	print('primacy:', sum(avg.avg('Primacy', exp=0)), sum(avg.avg('Primacy', exp=1)), sum(avg.avg('Primacy', exp=100)))
	print('recency:', sum(avg.avg('Recency', exp=0)), sum(avg.avg('Recency', exp=1)), sum(avg.avg('Recency', exp=100)))
	print('Gaussian:', sum(avg.avg('Gaussian', mu=mu, sigma=10)), sum(avg.avg('Gaussian', mu=mu, sigma=1)), sum(avg.avg('Gaussian', mu=mu, sigma=.1)))

if __name__ == "__main__":
	main()
