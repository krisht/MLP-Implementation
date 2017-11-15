import numpy as np
import os
from scipy.special import expit as sig

def dsig(X):
    return sig(X) * (1 - sig(X))


class MultiLayerPerceptron:
    def __init__(self,  arch):
        self.arch = arch
        self.weights = {}
        self.biases = {}

        self.y = None

        self.num_layers = len(arch) - 1

        for ii in range(self.num_layers):
            self.weights['w_%d_%d' % (ii, ii + 1)] = np.zeros(arch[ii:ii + 2])

    def forward_prop(self, x):
        if len(x.shape) != 2 or x.shape[1] != self.arch[0]:
            raise ValueError("Incorrect input shape " + str(x.shape) + " given!")
        else:
            self.y = None
            for ii in range(self.num_layers):
                if self.y is None:
                    self.y = sig(np.matmul(x, self.weights["w_%d_%d" % (ii, ii + 1)]))
                else:
                    self.y = sig(np.matmul(self.y, self.weights["w_%d_%d" % (ii, ii + 1)]))

        return self.y

    def load_initial_weights(self, file):

        pass


    def back_prop(self, trainX, trainY, gamma, lamduh):
        pass


    def test_network(self):
        pass

    def train_network(self):
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

    lambduh = float(input("Enter learning rate: "))
    while not lambduh > 0.0:
        lambduh = float(input("Enter learning rate: "))


if __name__ == "__main__":
    __train_neural_network__()
