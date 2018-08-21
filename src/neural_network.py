import numpy as np
import json
import costs
from activations import sigmoid
import sys

class NeuralNetwork:
    """
    A module to implement the stochastic gradient descent learning algorithm and the computation of gradients using backpropagation for a fully-connected neural network.
    """

    def __init__(self, layer_sizes, cost=costs.CrossEntropyCost):

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases to random values
        self.initialize_weights()

        # Define activation and cost function
        self.activation = sigmoid
        self.cost = cost

    def initialize_weights(self):
        """ Initializes the weights and biases with a Gaussian distribution so that they have a mean of 0 and a standard deviation of the inverse of number of neurons in the previous layer. """
        self.biases = [np.random.randn(n, 1) for n in self.layer_sizes[1:]]
        self.weights = [np.random.randn(n, m) / np.sqrt(m) for n, m in zip(self.layer_sizes[1:], self.layer_sizes[:-1])]

    def mini_batch_gradient_descent(self, training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, regularization_parameter=0, test_data=None):
        """
        Trains the neural network on the training data using mini-batch gradient descent.

        Parameters:
        ----------
        `training_data`: The training data that the model should be trained on.

        `epochs`: The number of iterations to train on all of the training data.

        `mini_batch_size`: The size of the mini-batches that gradient descent should take steps with. Larger mini-batch sizes correspond to more accurate gradients but longer training times.

        `learning_rate`: The learning rate controls how big the steps gradient descent takes are. If it is too small, the model will take too long to converge. If it is too large, it might overshoot the local minimum we are trying to reach.

        `regularization_parameter`: The regularization parameter controls how well the model generalizes to new data. If it is too small, the model will be likely to overfit the training data and perform badly on the test data. If it is too large, the model is likely to perform badly on both the training and the test data.
        
        `test_data`: The test data used to evaluate the model and print the progress made after each epoch.
        """
        
        # Number of rows in training data
        num_rows = len(training_data)

        for epoch in range(epochs):

            # Shuffle the training data
            np.random.shuffle(training_data)

            # Partition the data into mini-batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, num_rows, mini_batch_size)]

            # Update weights and biases for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, regularization_parameter)

            # Print progress after each epoch
            if test_data:
                print('Epoch {0}: {1}% accuracy'.format(epoch+1, round(self.accuracy(test_data)*100,2)))
            else:
                print('Epoch {0} completed'.format(epoch+1))

    def update_mini_batch(self, mini_batch, learning_rate, regularization_parameter):
        """
        Updates the weights and biases of the neural network by applying gradient descent to the mini-batch. The regularization method used is L2 regularization.
        
        Parameters
        ----------
        `mini_batch`: The mini-batch to train on.

        `learning_rate`: The learning rate to be used to update the weights and biases.
        """

        mini_batch_size = len(mini_batch)
        
        # Initialize empty gradient arrays for the batch
        W_gradient_batch = [np.zeros(W.shape) for W in self.weights]
        b_gradient_batch = [np.zeros(b.shape) for b in self.biases]

        # Iterate over each (x, y) pair in mini-batch
        for x, y in mini_batch:

            # Compute gradients of current (x, y) pair
            W_gradient, b_gradient = self.backprop(x, y)

            # Add gradients of current (x, y) pair to the batch gradient
            W_gradient_batch = [Wg_b + Wg for Wg_b, Wg in zip(W_gradient_batch, W_gradient)]
            b_gradient_batch = [bg_b + bg for bg_b, bg in zip(b_gradient_batch, b_gradient)]

        # Update weights and biases according to the learning rate, the regularization parameter and the computed mini-batch gradient
        self.weights = [W - ((learning_rate/mini_batch_size) * W_gradient) - ((learning_rate * regularization_parameter / mini_batch_size) * W) for W, W_gradient in zip(self.weights, W_gradient_batch)]
        self.biases = [b - ((learning_rate/mini_batch_size) * b_gradient) for b, b_gradient in zip(self.biases, b_gradient_batch)]

    def accuracy(self, test_data):
        """ Returns the prediction accuracy on the given test data. """
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

        # Initialize empty gradient arrays
        W_gradient = [np.zeros(W.shape) for W in self.weights]
        b_gradient = [np.zeros(b.shape) for b in self.biases]

        # Feedforward and store computed a and z values
        a_vals, z_vals = self.feedforward(x, return_vals=True)

        # Compute first layer of gradients using the derivative of the cost
        delta = self.cost.delta(z_vals[-1], a_vals[-1], y)
        W_gradient[-1] = np.dot(delta, a_vals[-2].T)
        b_gradient[-1] = delta
        
        # Propagate error backwards and compute gradients of all preceeding layers
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].T, delta) * self.activation(z_vals[-l], derivative=True)
            W_gradient[-l] = np.dot(delta, a_vals[-l-1].T)
            b_gradient[-l] = delta

        return W_gradient, b_gradient

    def predict(self, x):
        """ Returns the predicted label for the input image `x`. """
        return np.argmax(self.feedforward(x))

    def serialize(self, filename):
        """ Saves the neural network to the file `filename`. """

        # Save data as dictionary
        data = {"lazer_sizes": self.layer_sizes,
                "weights": self.weights,
                "biases": self.biases,
                "cost": str(self.cost.__name__)}

        # Write dictionary to file as json
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()

    @staticmethod
    def deserialize(filename):
        """ Loads and deserializes a serialized `NeuralNetwork` object from a file. """

        # Load data from json file
        f = open(filename, 'r')
        data = json.load(f)
        f.close()
        
        # Get cost class
        cost = getattr(sys.modules['costs'], data['cost'])

        # Intialize a new neural network instance with loaded weights, biases and cost
        network = NeuralNetwork(data['layer_sizes'], cost)
        network.weights = data['weights']
        network.biases = data['biases']

        return network

