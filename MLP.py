import os
import sys

import numpy as np
import sklearn.datasets
from scipy.special import expit as sig
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3)


class MultiLayerPerceptron:
	def __init__(self, init_weights_file=None, train_data_file=None, test_data_file=None, output_file=None,
				 num_epochs=None, lambdaa=None):
		self.train_data_file = train_data_file
		self.test_data_file = test_data_file
		self.init_weights_file = init_weights_file
		self.output_file = output_file
		self.num_epochs = num_epochs
		self.lambdaa = lambdaa
		self.arch = []
		self.weights = {}

		self.a1 = self.a2 = self.a3 = None
		self.ins2 = self.ins3 = None

		self.load_initial_weights()

		self.y = None

		self.num_layers = len(self.arch) - 1

		if self.train_data_file is not None:
			self.X_train, self.y_train = self.get_data(self.train_data_file)
		else:
			self.X_train, self.y_train = None, None

		if self.test_data_file is not None:
			self.X_test, self.y_test = self.get_data(self.test_data_file)
		else:
			self.X_test, self.y_test = None, None

		# self.train_network()

	def get_data(self, file_name):
		data = []
		with open(file_name, 'r') as file:
			temp = file.readline().split()
			num_inputs = int(temp[1])
			num_outputs = int(temp[2])

			if num_inputs != self.arch[0]:
				raise ValueError("Incorrect input dimensions.")
			elif num_outputs != self.arch[2]:
				raise ValueError("Incorrect output dimensions.")

			for line in file:
				line = [float(i) for i in line.split()]
				data = data + [line]
			x = np.asarray(data).T[:num_inputs].T
			y = np.asarray(data).T[num_inputs:].T
			return x, y
		return None, None

	def forward_prop(self, x):
		if len(x.shape) != 2 or x.shape[1] != self.arch[0]:
			raise ValueError("Incorrect input shape " + str(x.shape) + " given!")
		else:
			self.y = None
			x = np.append(-np.ones((len(x), 1)), x, axis=1)
			self.a1 = x
			self.ins2 = np.matmul(self.a1, self.weights["w1"])
			self.a2 = sig(self.ins2)
			self.a2 = np.append(-np.ones((len(self.a2), 1)), self.a2, axis=1)
			self.ins3 = np.matmul(self.a2, self.weights["w2"])
			self.a3 = sig(self.ins3)
			self.y = self.a3
		return self.y

	def load_initial_weights(self):
		with open(self.init_weights_file, 'r') as file:
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

			self.arch = arch
			self.weights['w1'] = w1
			self.weights['w2'] = w2

	def test_network(self):
		orig_results = np.zeros((self.arch[2], 4))
		y_pred = np.round(self.forward_prop(self.X_test), 0)
		for ii in range(self.arch[2]):
			orig_results[ii, :] = np.reshape(confusion_matrix(self.y_test[:, ii], y_pred[:, ii]), 4)
			orig_results[ii, 0], orig_results[ii, 3] = orig_results[ii, 3], orig_results[ii, 0]

		orig_results = np.asarray(orig_results, dtype=np.int32)

		accuracy_score, precision, recall, f1 = calculate_metrics(orig_results)

		results = np.concatenate((orig_results, np.expand_dims(accuracy_score, 0).T, np.expand_dims(precision, 0).T, np.expand_dims(recall, 0).T, np.expand_dims(f1, 0).T), 1)

		temp = np.average(results[:, 4:], axis=0)
		temp[3] = 2 * temp[1] * temp[2] / (temp[1] + temp[2])

		with open(self.output_file, 'w') as f:
			for ii in range(results.shape[0]):
				f.write("%d %d %d %d %0.3f %0.3f %0.3f %0.3f\n" % tuple(results[ii, :]))
			f.write("%0.3f %0.3f %0.3f %0.3f\n" % calculate_metrics(np.sum(orig_results, axis=0, keepdims=True)))
			f.write("%0.3f %0.3f %0.3f %0.3f\n" % tuple(temp))

	def train_network(self):

		for _ in range(self.num_epochs):
			for ii in range(len(self.X_train)):
				tempX = self.X_train[ii:ii + 1, :]
				tempY = self.y_train[ii:ii + 1, :]
				self.forward_prop(tempX)
				delta3 = dsig(self.ins3) * (tempY - self.a3)
				delta2 = dsig(self.ins2) * np.matmul(delta3, self.weights['w2'][1:, ].T)
				self.weights['w2'] = self.weights['w2'] + self.lambdaa * np.matmul(self.a2.T, delta3)
				self.weights['w1'] = self.weights['w1'] + self.lambdaa * np.matmul(self.a1.T, delta2)

		with open(self.output_file, 'w') as f:
			f.write('%d %d %d\n' % (self.arch[0], self.arch[1], self.arch[2]))

		np.savetxt(open(self.output_file, 'ab'), self.weights['w1'].T, '%0.3f', delimiter=' ')
		np.savetxt(open(self.output_file, 'ab'), self.weights['w2'].T, '%0.3f', delimiter=' ')


def __train_neural_network__():
	print("Neural Network Training Program")

	# Request text file with original weights
	initial_weights_path = input("Enter initial weight file location: ")
	while not os.path.isfile(initial_weights_path):
		initial_weights_path = input("Enter initial weight file location: ")

	# Request training set location
	training_set_path = input("Enter training set file location: ")
	while not os.path.isfile(training_set_path):
		training_set_path = input("Enter training set file location: ")

	# Request output file location
	output_path = input("Enter output file location: ")
	while not os.path.isfile(output_path):
		open(output_path, 'w')

	# Request number of epochs
	num_epochs = int(input("Enter positive integer for number of epochs: "))
	while not num_epochs > 0:
		num_epochs = int(input("Enter positive integer for number of epochs: "))

	# Request learning rate
	lambdaa = float(input("Enter learning rate: "))
	while not lambdaa > 0.0:
		lambdaa = float(input("Enter learning rate: "))

	net = MultiLayerPerceptron(init_weights_file=initial_weights_path, train_data_file=training_set_path,
							   test_data_file=None, output_file=output_path, num_epochs=num_epochs, lambdaa=lambdaa)
	net.train_network()


def __test_neural_network__():
	print("Neural Network Testing Program")

	# Request text file with original weights
	trained_weights = input("Enter trained neural network file: ")
	while not os.path.isfile(trained_weights):
		trained_weights = input("Enter trained neural network file: ")

	# Request text file with test set
	testing_set_path = input("Enter testing set file location: ")
	while not os.path.isfile(testing_set_path):
		testing_set_path = input("Enter testing set file location: ")

	# Ouput text file name
	output_path = input("Enter output file location: ")
	while not os.path.isfile(output_path):
		open(output_path, 'w')

	net = MultiLayerPerceptron(init_weights_file=trained_weights, test_data_file=testing_set_path, output_file=output_path)
	net.test_network()


def dsig(x):
	return sig(x) * (1 - sig(x))


def calculate_metrics(results):
	accuracy_score = np.asarray((results[:, 0] + results[:, 3]) / np.sum(results, 1), dtype=np.float32)
	precision = np.asarray((results[:, 0]) / np.sum(results[:, 0:2], 1), dtype=np.float32)
	recall = np.asarray((results[:, 0]) / (results[:, 0] + results[:, 2]), dtype=np.float32)
	f1 = np.asarray(2 * precision * recall / (precision + recall), dtype=np.float32)
	return accuracy_score, precision, recall, f1


def generate_dataset():
	n_features = 20
	n_hidden = 25
	n_classes = 5

	train_size = 400
	test_size = 100

	x, y = sklearn.datasets.make_multilabel_classification(train_size + test_size, n_features, n_classes, allow_unlabeled=False)
	x = x / 17.0
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

	with open('krishna.train', 'w') as f:
		f.write("%d %d %d\n" % (train_size, n_features, n_classes))

	with open('krishna.test', 'w') as f:
		f.write("%d %d %d\n" % (test_size, n_features, n_classes))

	train_set = np.concatenate((x_train, y_train), axis=1)
	test_set = np.concatenate((x_test, y_test), axis=1)

	format_string = "%0.3f " * n_features + "%d " * n_classes
	np.savetxt(open('krishna.train', 'ab'), train_set, format_string[:-1], delimiter=' ')
	np.savetxt(open('krishna.test', 'ab'), test_set, format_string[:-1], delimiter=' ')

	with open('krishna.init', 'w') as f:
		f.write('%d %d %d\n' % (n_features, n_hidden, n_classes))

	np.savetxt(open('krishna.init', 'ab'), np.random.rand(n_features + 1, n_hidden).T, '%0.3f', ' ')
	np.savetxt(open('krishna.init', 'ab'), np.random.rand(n_hidden + 1, n_classes).T, '%0.3f', ' ')



if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python MLP.py [train | test | gen_data]")
	elif sys.argv[1] == 'test':
		__test_neural_network__()
	elif sys.argv[1] == 'train':
		__train_neural_network__()
	elif sys.argv[1] == 'gen_data':
		generate_dataset()
	else:
		print("usage: python MLP.py [train | test | gen_data]")