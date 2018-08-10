import numpy as np

class NeuralNetwork:

    def __init__(self, layer_sizes):

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # initialize weights and biases to random values
        self.biases = [np.random.randn(n, 1) for n in layer_sizes[1:]]
        self.weights = [np.random.randn(n, m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

        # define activation and cost function
        self.activation = sigmoid
        self.cost = quadratic_cost

    def stochastic_gradient_descent(self, training_data, test_data, epochs, mini_batch_size, alpha):
        
        # number of rows in dataset
        m = len(training_data)

        for j in  range(epochs):

            # shuffle the training data
            np.random.shuffle(training_data)

            # partition the data into mini-batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, m, mini_batch_size)]

            # update weights and biases for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)

            # print progress after each epoch
            print('Epoch {0}: {1} / {2}'.format(j+1, self.evaluate(test_data), len(test_data)))

    def update_mini_batch(self, mini_batch, alpha):

        # mini-batch size
        n = len(mini_batch)
        
        # initialize empty gradient arrays for the batch
        W_gradient_batch = [np.zeros(W.shape) for W in self.weights]
        b_gradient_batch = [np.zeros(b.shape) for b in self.biases]

        # iterate over each (x, y) pair in mini-batch
        for x, y in mini_batch:

            # compute gradients of current (x, y) pair
            W_gradient, b_gradient = self.backprop(x, y)

            # add gradients of current (x, y) pair to the batch gradient
            W_gradient_batch = [Wg_b + Wg for Wg_b, Wg in zip(W_gradient_batch, W_gradient)]
            b_gradient_batch = [bg_b + bg for bg_b, bg in zip(b_gradient_batch, b_gradient)]

        # update weights and biases according to learning rate alpha and the computed batch gradient
        self.weights = [W - ((alpha/n) * W_gradient) for W, W_gradient in zip(self.weights, W_gradient_batch)]
        self.biases = [b - ((alpha/n) * b_gradient) for b, b_gradient in zip(self.biases, b_gradient_batch)]


    def evaluate(self, test_data): # return the number of correctly classified data points
        test_results = [(self.predict(x)==np.argmax(y)) for x, y in test_data]
        return sum(test_results)

    def feedforward(self, x, return_vals=False):

        if not return_vals:
            a = x
            for W, b in zip(self.weights, self.biases):
                a = self.activation(np.dot(W, a) + b)
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
            return a_vals, z_vals

    def backprop(self, x, y):

        # initialize empty gradient arrays
        W_gradient = [np.zeros(W.shape) for W in self.weights]
        b_gradient = [np.zeros(b.shape) for b in self.biases]

        # feedforward and store computed a and z values
        a_vals, z_vals = self.feedforward(x, return_vals=True)

        # compute first layer of gradients using the derivative of the cost
        #delta = np.multiply(self.cost(a_vals[-1], y, derivative=True), self.activation(z_vals[-1], derivative=True))
        delta = self.cost(a_vals[-1], y, derivative=True) * self.activation(z_vals[-1], derivative=True)
        W_gradient[-1] = np.dot(delta, a_vals[-2].T)
        b_gradient[-1] = delta
        
        # propagate error backwards and compute gradients of all preceeding layers
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * self.activation(z_vals[-l], derivative=True)
            W_gradient[-l] = np.dot(delta, a_vals[-l-1].T)
            b_gradient[-l] = delta

        return W_gradient, b_gradient

    def predict(self, x):
        return np.argmax(self.feedforward(x))

def sigmoid(z, derivative=False):
    if not derivative:
        return 1.0/(1.0 + np.exp(-z))
    else:
        return np.multiply(sigmoid(z), sigmoid(1-z))

def quadratic_cost(a, y, derivative=False):
    # sum of squares error
    if not derivative:
        return (1/2) * np.sum((a - y)**2)
    else:
        return (a - y)