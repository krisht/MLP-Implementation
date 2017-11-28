import os

import numpy as np
from scipy.special import expit as sig
from sklearn.metrics import confusion_matrix


#
# def dsig(X):
# 	return sig(X) * (1 - sig(X))

# def sig(X):
# 	return sig(X)

class MultiLayerPerceptron:
	def __init__(self, init_weights_file=None, train_data_file=None, test_data_file=None, num_epochs=None, lambdah=None):
		self.train_data_file = train_data_file
		self.test_data_file = test_data_file
		self.init_weights_file = init_weights_file
		self.num_epochs = num_epochs
		self.lambdah = lambdah
		self.arch = []
		self.weights = {}
		self.biases = {}

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
		#print(self.X_test)
		#print(self.y_test)

		#self.train_network()

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

	def forward_prop(self, x):
		if len(x.shape) != 2 or x.shape[1] != self.arch[0]:
			raise ValueError("Incorrect input shape " + str(x.shape) + " given!")
		else:
			self.y = None
			for ii in range(self.num_layers):
				if self.y is None:
					#print(sig(np.matmul(x, self.weights["w_%d_%d" % (ii, ii + 1)]) - self.biases["b_%d_%d" % (ii, ii + 1)]))
					self.y = sig(np.matmul(x, self.weights["w_%d_%d" % (ii, ii + 1)]) - self.biases["b_%d_%d" % (ii, ii + 1)])
				else:
					#print(sig(np.matmul(self.y, self.weights["w_%d_%d" % (ii, ii + 1)])  - self.biases["b_%d_%d" % (ii, ii + 1)]))
					self.y = sig(np.matmul(self.y, self.weights["w_%d_%d" % (ii, ii + 1)]) - self.biases["b_%d_%d" % (ii, ii + 1)])
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
		results = np.zeros((self.arch[2], 4))
		y_pred = np.round(self.forward_prop(self.X_test), 0)
		for ii in range(self.arch[2]):
			results[ii, :] = np.reshape(confusion_matrix(self.y_test[:, ii], y_pred[:, ii]), (4))
		overall_accuracy = (results[:, 0] + results[:, 3])/np.sum(results, 1)
		precision = (results[:, 0])/(results[:, 0] + results[:, 1])
		recall = (results[:, 0])/(results[:, 0] + results[:, 2])
		f1 = (2 * precision * recall)/(precision + recall)
		print(results[:,2])

		print(f1)


	def train_network(self):
		# for X, y in zip(self.X_train, self.y_train):
		# 	#print(X, y)
		pass


def __train_neural_network__():
	# print("Neural Network Training Program")

	# # Request text file with original weights
	# initial_weights_path = input("Enter initial weight file location: ")
	# while not os.path.isfile(initial_weights_path):
	# 	initial_weights_path = input("Enter initial weight file location: ")
	#
	# # Request
	# training_set_path = input("Enter training set file location: ")
	# while not os.path.isfile(training_set_path):
	# 	training_set_path = input("Enter training set file location: ")
	#
	# output_path = input("Enter output file location: ")
	# while not os.path.isfile(output_path):
	# 	output_path = input("Enter output file location: ")
	#
	# num_epochs = int(input("Enter positive integer for number of epochs: "))
	# while not num_epochs > 0:
	# 	num_epochs = int(input("Enter positive integer for number of epochs: "))
	#
	# lambdah = float(input("Enter learning rate: "))
	# while not lambdah > 0.0:
	# 	lambdah = float(input("Enter learning rate: "))

	num_epochs = 10
	lambdah = 5
	initial_weights_path = '/home/krishna/Dropbox/MultiLayerPerceptron/sampleB.trained'
	train_data_file_path = '/home/krishna/Dropbox/MultiLayerPerceptron/sampleB.train'
	test_data_file_path = '/home/krishna/Dropbox/MultiLayerPerceptron/sampleB.test'

	net = MultiLayerPerceptron(init_weights_file=initial_weights_path, train_data_file=train_data_file_path,
								test_data_file=test_data_file_path, num_epochs=num_epochs, lambdah=lambdah)
	net.test_network()


def __test_neural_network__():
	# print("Neural Network Testing Program")

	# Request text file with original weights
	trained_weights = input("Enter trained neural network file: ")
	while not os.path.isfile(trained_weights):
		trained_weights = input("Enter trained neural network file: ")

	# Request
	testing_set_path = input("Enter testing set file location: ")
	while not os.path.isfile(testing_set_path):
		testing_set_path = input("Enter testing set file location: ")

	output_path = input("Enter output file location: ")
	while not os.path.isfile(output_path):
		output_path = input("Enter output file location: ")


if __name__ == "__main__":
	__train_neural_network__()
