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
				 num_epochs=None, lambdah=None):
		self.train_data_file = train_data_file
		self.test_data_file = test_data_file
		self.init_weights_file = init_weights_file
		self.output_file = output_file
		self.num_epochs = num_epochs
		self.lambdah = lambdah
		self.arch = []
		self.weights = {}
		self.biases = {}

		self.inputs = []
		self.activations = []
		self.zs = []
		self.outputs = []

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
			self.inputs = []
			self.outputs = []
			self.zs = []
			self.activations = []
			for ii in range(self.num_layers):
				if self.y is None:
					self.inputs = self.inputs + [x]
					self.zs = self.zs + [
						np.matmul(x, self.weights["w_%d_%d" % (ii, ii + 1)]) - self.biases["b_%d_%d" % (ii, ii + 1)]]
					self.y = sig(
						np.matmul(x, self.weights["w_%d_%d" % (ii, ii + 1)]) - self.biases["b_%d_%d" % (ii, ii + 1)])
					self.outputs = self.outputs + [self.y]
				else:
					self.inputs = self.inputs + [self.y]
					self.zs = self.zs + [np.matmul(self.y, self.weights["w_%d_%d" % (ii, ii + 1)]) - self.biases[
						"b_%d_%d" % (ii, ii + 1)]]
					self.y = sig(np.matmul(self.y, self.weights["w_%d_%d" % (ii, ii + 1)]) - self.biases[
						"b_%d_%d" % (ii, ii + 1)])
					self.outputs = self.outputs + [self.y]
		return self.y

	def load_initial_weights(self):
		with open(self.init_weights_file, 'r') as file:
			arch = [int(i) for i in file.readline().split()]

			weights = []
			for ii in range(arch[1]):
				line = file.readline()
				line = [float(i) for i in line.split()]
				weights = weights + [line]
			w_0_1 = np.asarray(weights).T

			weights = []
			for ii in range(arch[2]):
				line = file.readline()
				line = [float(i) for i in line.split()]
				weights = weights + [line]
			w_1_2 = np.asarray(weights).T

			self.arch = arch
			self.biases['b_0_1'] = w_0_1[0]
			self.weights['w_0_1'] = w_0_1[1:]
			self.biases['b_1_2'] = w_1_2[0]
			self.weights['w_1_2'] = w_1_2[1:]

	def back_prop(self, gamma, lamduh):
		pass

	def test_network(self):
		orig_results = np.zeros((self.arch[2], 4))
		y_pred = np.round(self.forward_prop(self.X_test), 0)
		for ii in range(self.arch[2]):
			orig_results[ii, :] = np.reshape(confusion_matrix(self.y_test[:, ii], y_pred[:, ii]), 4)
			orig_results[ii, 0], orig_results[ii, 3] = orig_results[ii, 3], orig_results[ii, 0]

		orig_results = np.asarray(orig_results, dtype=np.int32)

		accuracy_score, precision, recall, f1 = calculate_metrics(orig_results)

		results = np.concatenate((orig_results, np.expand_dims(accuracy_score, 0).T, np.expand_dims(precision, 0).T,
								  np.expand_dims(recall, 0).T, np.expand_dims(f1, 0).T), 1)

		temp = np.average(results[:, 4:], axis=0)
		temp[3] = 2 * temp[1] * temp[2] / (temp[1] + temp[2])

		with open(self.output_file, 'w') as f:
			for ii in range(results.shape[0]):
				f.write("%d %d %d %d %0.3f %0.3f %0.3f %0.3f\n" % tuple(results[ii, :]))
			f.write("%0.3f %0.3f %0.3f %0.3f\n" % calculate_metrics(np.sum(orig_results, axis=0, keepdims=True)))
			f.write("%0.3f %0.3f %0.3f %0.3f\n" % tuple(temp))

	def train_network(self):
		y_pred = self.forward_prop(self.X_train)
		delj1 = np.matmul(dsig(self.inputs[1]).T, (self.y_train - self.outputs[1]))
		delj2 = np.matmul(dsig(self.inputs[0]).T, self.zs[0])

		pass


def __train_neural_network__():
	print("Neural Network Training Program")

	# Request text file with original weights
	initial_weights_path = input("Enter initial weight file location: ")
	while not os.path.isfile(initial_weights_path):
		initial_weights_path = input("Enter initial weight file location: ")

	# Request
	training_set_path = input("Enter training set file location: ")
	while not os.path.isfile(training_set_path):
		training_set_path = input("Enter training set file location: ")

	output_path = input("Enter output file location: ")
	while not os.path.isfile(output_path):
		output_path = input("Enter output file location: ")

	num_epochs = int(input("Enter positive integer for number of epochs: "))
	while not num_epochs > 0:
		num_epochs = int(input("Enter positive integer for number of epochs: "))

	lambdah = float(input("Enter learning rate: "))
	while not lambdah > 0.0:
		lambdah = float(input("Enter learning rate: "))

	num_epochs = 10
	lambdah = 5
	initial_weights_path = '/home/krishna/Dropbox/MultiLayerPerceptron/sampleA.trained'
	train_data_file_path = '/home/krishna/Dropbox/MultiLayerPerceptron/sampleA.train'
	test_data_file_path = '/home/krishna/Dropbox/MultiLayerPerceptron/sampleA.test'
	output_path = 'blah.txt'
	net = MultiLayerPerceptron(init_weights_file=initial_weights_path, train_data_file=train_data_file_path,
							   test_data_file=test_data_file_path, output_file='blah.txt', num_epochs=num_epochs,
							   lambdah=lambdah)
	net.test_network()


def __test_neural_network__():
	print("Neural Network Testing Program")

	# Request text file with original weights
	trained_weights = input("Enter trained neural network file: ")
	while not os.path.isfile(trained_weights):
		trained_weights = input("Enter trained neural network file: ")

	# Request
	testing_set_path = input("Enter testing set file location: ")
	while not os.path.isfile(testing_set_path):
		testing_set_path = input("Enter testing set file location: ")

	output_path = input("Enter output file location: ")

	net = MultiLayerPerceptron(init_weights_file=trained_weights, test_data_file=testing_set_path,
							   output_file=output_path)
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
	n_classes = 5

	train_size = 400
	test_size = 100

	x, y = sklearn.datasets.make_multilabel_classification(train_size + test_size, n_features, n_classes,
														   allow_unlabeled=False)
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

	with open('krishna_dataset.train', 'w') as f:
		f.write("%d %d %d\n" % (train_size, n_features, n_classes))

	with open('krishna_dataset.test', 'w') as f:
		f.write("%d %d %d\n" % (test_size, n_features, n_classes))

	train_set = np.concatenate((x_train, y_train), axis=1)
	test_set = np.concatenate((x_test, y_test), axis=1)
	np.savetxt(open('krishna_dataset.train', 'ab'), train_set, '%d', delimiter=' ')
	np.savetxt(open('krishna_dataset.test', 'ab'), test_set, '%d', delimiter=' ')

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python MLP.py <train | test>")
	elif sys.argv[1] == 'test':
		__test_neural_network__()
	elif sys.argv[1] == 'train':
		__train_neural_network__()
	elif sys.argv[1] == 'gen_data':
		generate_dataset()
	else:
		print("usage: python MLP.py <train | test | gen_data>")

generate_dataset()
