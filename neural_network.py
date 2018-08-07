import numpy as np

class NeuralNetwork:

    def __init__(self, layer_sizes):

        sample_layer_sizes = (784, 16, 16, 10) # possible MNIST architecture

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # initialize weights and biases to random values
        self.biases = [np.randn(n, 1) for n in layer_sizes[1:]]
        self.weights = [np.randn(n, m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

        # define activation and cost function
        self.activation = sigmoid
        self.cost = quadratic_cost

    def stochastic_gradient_descent(self, X, y, epochs, mini_batch_size, alpha):
        m = len(X)

    def feedforward(self, x, return_vals=False):

        if not return_vals:
            a = x
            for W, b in zip(self.weights, self.biases):
                a = sigmoid(np.dot(W, a) + b)
            return a
            
        else:
            a = x
            z_vals = []
            a_vals = [x]
            for W, b in zip(self.weights, self.biases):
                z = np.dot(W, a) + b
                z_vals.append(z)
                a = self.activation(z)
                a_vals.append(a)
            return z_vals, a_vals

    def backprop(self, x, y):

        # initialize empty gradient arrays
        W_gradient = [np.zeros(W.shape) for W in self.weights]
        b_gradient = [np.zeros(b.shape) for b in self.biases]

        # feedforward and store intermediate values
        z_vals, a_vals = self.feedforward(x, return_vals=True)

        # compute first layer of gradients using the derivative of the cost
        delta = np.multiply(cost(a_vals[-1], y, derivative=True), activation(z_vals[-1], derivative=True))
        W_gradient[-1] = np.dot(delta, a_vals[-2].transpose()) # consider adding this eqn to the .tex
        b_gradient[-1] = delta
        
        # propagate error backwards and compute gradients of all preceeding layers
        for l in range(2, self.num_layers):
            delta = np.multiply(np.dot(self.weights[-l+1].transpose(), delta), activation(z_vals[-l], derivative=True))
            W_gradient[-l] = np.dot(delta, a_vals[-l-1].transpose())
            b_gradient[-l] = delta

        return W_gradient, b_gradient

    def predict(self, x):
        return np.argmax(self.feedforward(x))

def sigmoid(x, derivative=False):
    return np.multiply(sigmoid(x), sigmoid(1-x)) if derivative else 1/(1 + np.exp(-x))

def quadratic_cost(a, y, derivative=False):
    # sum of squares error
    if not derivative:
        return (1/2) * np.sum((a - y)**2)
    else:
        return (a - y)