import numpy as np
from scipy.special import expit as sigmoid


# noinspection PyPep8Naming
class MultiLayerPerceptron:
    def __init__(self, arch):
        self.arch = arch
        self.weights = {}
        self.biases = {}

        self.y = None

        self.num_layers = len(arch) - 1

        for ii in range(self.num_layers):
            self.weights['w_%d_%d' % (ii, ii + 1)] = np.random.randint(0, 5, arch[ii:ii + 2])

    def forward_prop(self, X):
        if len(X.shape) != 2 or X.shape[1] != self.arch[0]:
            raise ValueError("Incorrect input shape " + str(X.shape) + " given!")
        else:
            self.y = None
            for ii in range(self.num_layers):
                if self.y is None:
                    self.y = sigmoid(np.matmul(X, self.weights["w_%d_%d" % (ii, ii + 1)]))
                else:
                    self.y = sigmoid(np.matmul(self.y, self.weights["w_%d_%d" % (ii, ii + 1)]))

        return self.y

    def back_prop(self):
        pass

    def test_network(self):
        pass

    def train_network(self):
        pass


if __name__ == "__main__":
    mlp = MultiLayerPerceptron([3, 2, 3, 4, 5])
    print(mlp.forward_prop(np.random.randint(0, 5, (4, 3))))
