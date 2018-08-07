import numpy as np

class NeuralNetwork:

    def __init__(self, layer_sizes):
        sample_size = (784, 16, 16, 10)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.biases = [np.randn(n, 1) for n in layer_sizes[1:]]
        self.weights = [np.randn(n, m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

    def stochastic_gradient_descent(self, X, y, epochs, mini_batch_size, alpha):
        m = len(X)

    def feedforward(self, x):
        a = x
        for W, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(W, a) + b)
        return a

    def backprop(self, x, y):
        
        # make choice of cost and activation function
        cost = quadratic_cost
        activation = sigmoid

        # initialize empty gradient arrays
        W_gradient = [np.zeros(W.shape) for W in self.weights]
        b_gradient = [np.zeros(b.shape) for b in self.biases]

        # feedforward and store intermediate values
        a = x
        z_vals = []
        a_vals = [a]
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, x) + b
            z_vals.append(z)
            a = activation(z)
            a_vals.append(a_vals)

        # propagate backwards to compute gradients
        delta = cost(a_vals[-1], y, derivative=True) * activation(z_vals[-1], derivative=True)
        W_gradient[-1] = np.dot(delta, a_vals[-2].transpose())
        b_gradient[-1] = delta

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