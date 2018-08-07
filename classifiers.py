import numpy as np
class ConvolutionalNeuralNetwork:

    def __init__(self, patch_size, patch_stride):
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def fit(self, X, y):
        pass

    def relu(self, x):
        return np.max(0, x)

    def sigmoid(self, X, derivative=False):
        return np.multiply(self.sigmoid(X), self.sigmoid(1-X)) if derivative else 1/(1 + np.exp(-X))

    def flatten(self, X):
        pass

    def maxpool(self, X):
        pass

    def convolution(self, X):
        pass

    def dense(self, X): # fully connected layer
        pass

    def dropout(self, X):
        pass

class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes

    def fit(self, X, y):
        # unroll matrix to vector
        X = X.ravel()

        # initialize weigths to random values
        self.layer_sizes = len(X) + self.layer_sizes + len(y)
        self.W = list()
        for i in range(len(self.layer_sizes)-1):
            W_i = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
            self.W.append(W_i)

    def sigmoid(self, X, derivative=False):
        return np.multiply(self.sigmoid(X), self.sigmoid(1-X)) if derivative else 1/(1 + np.exp(-X))

    def propagate_forward(self, X):
        activation = self.sigmoid
        A = X
        for i in range(len(self.layer_sizes)-1):
            Z = np.matmul(self.W[i], A)
            A = activation(Z)
        return A

    def propagate_backward(self, A):
        pass

    def predict(self, X):
        A = self.propagate_forward(X)
        return np.argmax(A)