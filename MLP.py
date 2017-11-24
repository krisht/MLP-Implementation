import numpy as np
import os
from scipy.special import expit as sig
#
# def dsig(X):
# 	return sig(X) * (1 - sig(X))

# def sig(X):
# 	return sig(X)

class MultiLayerPerceptron:
	def __init__(self,  init_weights_file=None, data_file=None,  epoch=None, lambduh=None):
		self.data_file = data_file
		self.init_weights_file = init_weights_file
		self.num_epochs = epoch
		self.lambduh = lambduh
		self.arch = []
		self.weights = {}
		self.biases = {}

		self.load_initial_weights()

		self.y = None

		self.num_layers = len(self.arch) - 1

		self.X_train, self.y_train = self.get_data()

	def get_data(self):

		data = []
		with open(self.data_file, 'r') as file:
			temp = file.readline().split()
			num_samples = int(temp[0])
			num_inputs = int(temp[1])
			num_outputs = int(temp[2])

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
					self.y = sig(np.matmul(x, self.weights["w_%d_%d" % (ii, ii + 1)]))
					self.y = self.y - self.biases["b_%d_%d" % (ii, ii + 1)]
				else:
					self.y = sig(np.matmul(self.y, self.weights["w_%d_%d" % (ii, ii + 1)]))
					self.y = self.y - self.biases["b_%d_%d" % (ii, ii + 1)]

		print(self.y.shape)
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
			self.biases['b_0_1'] = w_0_1[-1]
			self.weights['w_0_1'] = w_0_1[0:-1]
			self.biases['b_1_2'] = w_1_2[-1]
			self.weights['w_1_2'] = w_1_2[0:-1]


	def back_prop(self, trainX, trainY, gamma, lamduh):
		pass


	def test_network(self):
		pass

	def train_network(self):
		pass


def __train_neural_network__():
	#print("Neural Network Training Program")

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
	# lambduh = float(input("Enter learning rate: "))
	# while not lambduh > 0.0:
	# 	lambduh = float(input("Enter learning rate: "))

	initial_weights_path = '/home/krishna/Dropbox/MultiLayerPerceptron/sampleB.trained'
	num_epochs = 10
	lambduh = 5
	data_file_path = '/home/krishna/Dropbox/MultiLayerPerceptron/sampleB.train'

	net = MultiLayerPerceptron(initial_weights_path, data_file_path, num_epochs, lambduh)
	rand_input = np.random.rand(50, 30)*100
	#print(rand_input)
	print(net.forward_prop(net.X_train))


def __test_neural_network__():
	#print("Neural Network Testing Program")

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

