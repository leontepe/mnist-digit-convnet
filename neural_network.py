import numpy as np

class NeuralNetwork:
    """
    A module to implement the stochastic gradient descent learning algorithm and the computation of gradients using backpropagation for a fully-connected neural network.
    """

    def __init__(self, layer_sizes):

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # initialize weights and biases to random values
        self.biases = [np.random.randn(n, 1) for n in layer_sizes[1:]]
        self.weights = [np.random.randn(n, m) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

        # define activation and cost function
        self.activation = sigmoid
        self.cost = quadratic_cost

    def stochastic_gradient_descent(self, training_data, epochs=10, mini_batch_size=10, learning_rate=1e-2, test_data=None):
        """
        Trains the neural network on the training data using stochastic gradient descent.

        Parameters:
        ----------
        `training_data`: The training data that the model should be trained on.

        `test_data`: The test data used to evaluate the model.

        `epochs`: The number of epochs.

        `mini_batch_size`: The size of the mini-batches.

        `learning_rate`: The learning rate.
        """
        
        # number of rows in training data
        num_rows = len(training_data)

        for epoch in range(epochs):

            # shuffle the training data
            np.random.shuffle(training_data)

            # partition the data into mini-batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, num_rows, mini_batch_size)]

            # update weights and biases for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            # print progress after each epoch
            if test_data:
                print('Epoch {0}: {1}% accuracy'.format(epoch+1, round(self.accuracy(test_data)*100,2)))
            else:
                print('Epoch {0} completed'.format(epoch+1))

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        Updates the weights and biases of the neural network by applying gradient descent to a mini-batch.
        
        Parameters
        ----------
        `mini_batch`: The mini-batch to train on.

        `learning_rate`: The learning rate to be used to update the weights and biases.
        """

        mini_batch_size = len(mini_batch)
        
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
        self.weights = [W - ((learning_rate/mini_batch_size) * W_gradient) for W, W_gradient in zip(self.weights, W_gradient_batch)]
        self.biases = [b - ((learning_rate/mini_batch_size) * b_gradient) for b, b_gradient in zip(self.biases, b_gradient_batch)]

    def accuracy(self, test_data):
        """ Returns the prediction accuracy on the give test data. """
        test_results = [(self.predict(x)==np.argmax(y)) for x, y in test_data]
        return sum(test_results) / len(test_data)

    def feedforward(self, x, return_vals=False):
        """
        Returns the output of the neural network given an input `x`.

        Parameters
        ----------
        `x`: The input to the neural network.

        `return_vals` (optional): `False` if just the activations of the output layer should be returned. `True` if all the intermediate a-values and  z-values should be returned as well.
        """

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
        """
        Returns the gradients of the cost w.r.t. each individual weight given a single `x` and `y` pair by using backpropagation.
        """

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
        """ Returns the predicted label for the input image `x`. """
        return np.argmax(self.feedforward(x))

def sigmoid(z, derivative=False):
    """ The sigmoid activation function and its derivative. """ 
    if not derivative:
        return 1.0/(1.0 + np.exp(-z))
    else:
        return np.multiply(sigmoid(z), sigmoid(1-z))

def quadratic_cost(a, y, derivative=False):
    """ The sum of squares error and its derivative. """
    if not derivative:
        return (1/2) * np.sum((a - y)**2)
    else:
        return (a - y)