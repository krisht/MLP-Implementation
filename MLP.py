import numpy as np
import os
from scipy.special import expit as sig

def dsig(X):
    return sig(X) * (1 - sig(X))


class MultiLayerPerceptron:
    def __init__(self,  input_file=None, epoch=None, lambduh=None):
        self.num_epochs = epoch
        self.lambduh = lambduh
        self.weights = {}
        self.biases = {}
        arch = [0, 0, 0]

        # with open(input_file, 'r') as file:
        #     arch = [int(i) for i in file.readline().split()]
        #     weights = []
        #     for line in file:
        #         line = [float(i) for i in line.split()]
        #         weights = weights + line
        #
        #     W * x + b
        #
        #
        #     self.weights['w_0_1'] = np.reshape(np.asarray(weights[:(arch[0])*(arch[1])]), (arch[0], arch[1]))
        #     self.biases['b_0_1'] = np.reshape(np.asarray(weights[(arch[0])*(arch[1]): (arch[0])*(arch[1] + 1)]), (1, arch[1]))
        #     self.weights['w_1_2'] = np.reshape(np.asarray(weights[(arch[1] + 1) * arch[2] : ]), (arch[1] + 1, arch[2]))
        #     self.biases['b_1_2'] = np.reshape(np.asarray(weights[(arch[1])*(arch[2]): (arch[1]+1)*(arch[2])]), (1, arch[2]))
        #
        #
        #     #print(len(weights[(arch[0] + 1)*(arch[1]):]))
        #
        #     print(self.weights['w_0_1'].shape)
        #     print(self.weights['w_1_2'].shape)
        #
        #     print(self.biases['b_0_1'].shape)
        #     print(self.biases['b_1_2'].shape)

    #     self.arch = arch
    #     self.weights = {}
    #     self.biases = {}
    #
    #     self.y = None
    #
    #     self.num_layers = len(arch) - 1
    #
    #     for ii in range(self.num_layers):
    #         self.weights['w_%d_%d' % (ii, ii + 1)] = np.zeros(arch[ii:ii + 2])
    #
    # def forward_prop(self, x):
    #     if len(x.shape) != 2 or x.shape[1] != self.arch[0]:
    #         raise ValueError("Incorrect input shape " + str(x.shape) + " given!")
    #     else:
    #         self.y = None
    #         for ii in range(self.num_layers):
    #             if self.y is None:
    #                 self.y = sig(np.matmul(x, self.weights["w_%d_%d" % (ii, ii + 1)]))
    #             else:
    #                 self.y = sig(np.matmul(self.y, self.weights["w_%d_%d" % (ii, ii + 1)]))
    #
    #     return self.y
    #
    # def load_initial_weights(self, file):
    #
    #     pass
    #
    #
    # def back_prop(self, trainX, trainY, gamma, lamduh):
    #     pass
    #
    #
    # def test_network(self):
    #     pass
    #
    # def train_network(self):
    #     pass


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

    lambduh = float(input("Enter learning rate: "))
    while not lambduh > 0.0:
        lambduh = float(input("Enter learning rate: "))


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
    while not os.path.isfile(output_path):
        output_path = input("Enter output file location: ")


# if __name__ == "__main__":
#     __train_neural_network__()


MultiLayerPerceptron(input_file='./sample.init')