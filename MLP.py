import os
import sys

import numpy as np
from scipy.special import expit as sig
from sklearn.metrics import confusion_matrix

try:
	assert sys.version_info >= (3,)
except AssertionError:
	sys.stdout.write('Run with python3...\n')
	sys.stdout.write('usage: python3 MLP.py [train|test]\n')
	sys.exit(-1)

np.set_printoptions(precision=3)


class MLP:
	def __init__(self, weights_file=None, train_file=None, test_file=None, output_file=None, n_epochs=None, alpha=None):
		self.train_file = train_file
		self.test_file = test_file
		self.weights_file = weights_file
		self.output_file = output_file
		self.n_epochs = n_epochs
		self.alpha = alpha
		self.n_inputs = None
		self.n_hidden = None
		self.n_output = None
		self.w1 = self.w2 = None

		self.a1 = self.a2 = self.a3 = None
		self.ins2 = self.ins3 = None

		self.load_initial_weights()

		if self.train_file is not None:
			self.X_train, self.y_train = self.get_data(self.train_file)
		else:
			self.X_train, self.y_train = None, None

		if self.test_file is not None:
			self.X_test, self.y_test = self.get_data(self.test_file)
		else:
			self.X_test, self.y_test = None, None

	def get_data(self, file_name):
		data = []
		with open(file_name, 'r') as file:
			temp = file.readline().split()
			n_inputs = int(temp[1])
			n_outputs = int(temp[2])

			if n_inputs != self.n_inputs:
				raise ValueError('Incorrect input dimensions.')
			elif n_outputs != self.n_output:
				raise ValueError('Incorrect output dimensions.')

			for line in file:
				line = [float(i) for i in line.split()]
				data = data + [line]

			x = np.asarray(data).T[:n_inputs].T
			y = np.asarray(data).T[n_inputs:].T

			return x, y
		return None, None

	def forward_prop(self, x):
		if len(x.shape) != 2 or x.shape[1] != self.n_inputs:
			raise ValueError('Incorrect input shape ' + str(x.shape) + ' given!')
		else:
			x = np.append(-np.ones((len(x), 1)), x, axis=1)
			self.a1 = x
			self.ins2 = np.matmul(self.a1, self.w1)
			self.a2 = sig(self.ins2)
			self.a2 = np.append(-np.ones((len(self.a2), 1)), self.a2, axis=1)
			self.ins3 = np.matmul(self.a2, self.w2)
			self.a3 = sig(self.ins3)
		return self.a3

	def load_initial_weights(self):
		with open(self.weights_file, 'r') as file:
			arch = [int(i) for i in file.readline().split()]

			weights = []
			for ii in range(arch[1]):
				line = file.readline()
				line = [float(i) for i in line.split()]
				weights = weights + [line]
			w1 = np.asarray(weights).T

			weights = []
			for ii in range(arch[2]):
				line = file.readline()
				line = [float(i) for i in line.split()]
				weights = weights + [line]
			w2 = np.asarray(weights).T

			self.n_inputs = arch[0]
			self.n_hidden = arch[1]
			self.n_output = arch[2]
			self.w1 = w1
			self.w2 = w2

	def test_network(self):
		orig_results = np.zeros((self.n_output, 4))
		y_hat = np.round(self.forward_prop(self.X_test), 0)
		for ii in range(self.n_output):
			orig_results[ii, :] = np.reshape(confusion_matrix(self.y_test[:, ii], y_hat[:, ii]), 4)
			orig_results[ii, 0], orig_results[ii, 3] = orig_results[ii, 3], orig_results[ii, 0]

		orig_results = np.asarray(orig_results, dtype=np.int32)
		accuracy, precision, recall, f1 = calculate_metrics(orig_results)

		accuracy = np.expand_dims(accuracy, 0).T
		precision = np.expand_dims(precision, 0).T
		recall = np.expand_dims(recall, 0).T
		f1 = np.expand_dims(f1, 0).T

		results = np.concatenate((orig_results, accuracy, precision, recall, f1), 1)

		temp = np.average(results[:, 4:], axis=0)
		temp[3] = 2 * temp[1] * temp[2] / (temp[1] + temp[2])

		with open(self.output_file, 'wb') as f:
			for ii in range(results.shape[0]):
				tmp_str = '%d %d %d %d %0.3f %0.3f %0.3f %0.3f\n' % tuple(results[ii, :])
				f.write(tmp_str.encode('utf-8'))
			tmp_str = '%0.3f %0.3f %0.3f %0.3f\n' % calculate_metrics(np.sum(orig_results, axis=0, keepdims=True))
			f.write(tmp_str.encode('utf-8'))
			tmp_str = '%0.3f %0.3f %0.3f %0.3f\n' % tuple(temp)
			f.write(tmp_str.encode('utf-8'))

		# Return value not used in most cases
		return np.average(accuracy)

	def train_network(self):

		for _ in range(self.n_epochs):
			for ii in range(len(self.X_train)):
				temp_x = self.X_train[ii:ii + 1, :]
				temp_y = self.y_train[ii:ii + 1, :]
				self.forward_prop(temp_x)
				delta3 = dsig(self.ins3) * (temp_y - self.a3)
				delta2 = dsig(self.ins2) * np.matmul(delta3, self.w2[1:, ].T)
				self.w2 += self.alpha * np.matmul(self.a2.T, delta3)
				self.w1 += self.alpha * np.matmul(self.a1.T, delta2)

		with open(self.output_file, 'wb') as f:
			tmp_str = '%d %d %d\n' % (self.n_inputs, self.n_hidden, self.n_output)
			f.write(tmp_str.encode('utf-8'))

		np.savetxt(open(self.output_file, 'ab'), self.w1.T, '%0.3f', delimiter=' ')
		np.savetxt(open(self.output_file, 'ab'), self.w2.T, '%0.3f', delimiter=' ')


def __train_neural_network__():
	print('Neural Network Training Program')

	# Request text file with original weights
	weights_file = input('Enter initial weight file location: ')
	while not os.path.isfile(weights_file):
		weights_file = input('Enter initial weight file location: ')

	# Request training set location
	train_path = input('Enter training set file location: ')
	while not os.path.isfile(train_path):
		train_path = input('Enter training set file location: ')

	# Request output file location
	output_file = input('Enter output file location: ')
	while not os.path.isfile(output_file):
		open(output_file, 'wb')

	# Request number of epochs
	n_epochs = int(input('Enter positive integer for number of epochs: '))
	while not n_epochs > 0:
		n_epochs = int(input('Enter positive integer for number of epochs: '))

	# Request learning rate
	alpha = float(input('Enter positive number for learning rate: '))
	while not alpha > 0.0:
		alpha = float(input('Enter positive number for learning rate: '))

	net = MLP(weights_file=weights_file, train_file=train_path, output_file=output_file, n_epochs=n_epochs, alpha=alpha)
	net.train_network()


def __test_neural_network__():
	print('Neural Network Testing Program')

	# Request text file with original weights
	trained_weights = input('Enter trained neural network file: ')
	while not os.path.isfile(trained_weights):
		trained_weights = input('Enter trained neural network file: ')

	# Request text file with test set
	test_file = input('Enter testing set file location: ')
	while not os.path.isfile(test_file):
		test_file = input('Enter testing set file location: ')

	# Output text file name
	output_file = input('Enter output file location: ')
	while not os.path.isfile(output_file):
		open(output_file, 'wb')

	net = MLP(weights_file=trained_weights, test_file=test_file, output_file=output_file)
	net.test_network()


def dsig(x):
	return sig(x) * (1 - sig(x))


def calculate_metrics(results):
	accuracy = np.asarray((results[:, 0] + results[:, 3]) / np.sum(results, 1), dtype=np.float32)
	precision = np.asarray((results[:, 0]) / np.sum(results[:, 0:2], 1), dtype=np.float32)
	recall = np.asarray((results[:, 0]) / (results[:, 0] + results[:, 2]), dtype=np.float32)
	f1 = np.asarray(2 * precision * recall / (precision + recall), dtype=np.float32)
	return accuracy, precision, recall, f1


try:
	if __name__ == '__main__':
		if len(sys.argv) < 2:
			print('usage: python3 MLP.py [train|test]')
		elif sys.argv[1] == 'test':
			__test_neural_network__()
		elif sys.argv[1] == 'train':
			__train_neural_network__()
		else:
			print('usage: python3 MLP.py [train|test]')
except IOError as io:
	sys.stdout.write('\nError opening file %s: %s. Exiting...\n' % (io.filename, io.strerror))
	sys.exit(-1)
except KeyboardInterrupt:
	sys.stdout.write('\nEncountered Ctrl + C. Exiting...\n')
	sys.exit(0)
