import matplotlib.pyplot as plt
import numpy as np
import random
import mnist
from neural_network import NeuralNetwork
import sys
import time
import os.path
import activations
import costs

def main():
    """ Main entry point of the script. """

    # settings
    load_network = True
    filename = 'network.json'
    epochs = 0
    plot_failures = False

    # load data
    print('Loading data...', end=' ')
    start = time.time()
    training_data, test_data = mnist.load_data()
    end = time.time()
    print('Finished')
    print('Time elapsed: %.2fs' % (round(end-start, 2)))
    
    # deserialize existing network (optional)
    if load_network and os.path.exists(filename):
        print('Found an existing model. Deserializing...', end=' ')
        model = NeuralNetwork.deserialize(filename)
        print('Finished')
    else:
        print('Initializing new model...', end=' ')
        model = NeuralNetwork([784, 30, 10], cost=costs.QuadraticCost)
        print('Finished')
    
    # train the network
    if epochs > 0:
        print('Training model...')
        start = time.time()
        model.mini_batch_gradient_descent(training_data, epochs=epochs, mini_batch_size=10, learning_rate=3, regularization_parameter=0, test_data=test_data)
        end = time.time()
        print('Time elapsed: %.2fs' % (round(end-start, 2)))

        print('Serializing model...', end=' ')
        model.serialize(filename)
        print('Finished')

    # plot misclassified test examples (optional)
    if plot_failures:
        print('Plotting misclassified images...')
        find_failures(model, test_data)

    # test np.argmax axis behaviour
    X = [[1, 3, 4], [2, 3, 1], [4, 0, 4], [0, 2, 5]]
    print(X, np.argmax(X, axis=1))

    # test if the neural network works with a batch-based approach
    n = 1000
    np.random.shuffle(training_data)
    X = np.array([np.array(x).flatten() for x, y in test_data])
    Y = np.array([np.array(y).flatten() for x, y in test_data])
    
    Y_pred = model.predict_batch(X[:n][:])
    Y_true = np.argmax(Y[:n][:], axis=1)
    print(sum(Y_pred==Y_true)/n)

def sample(data):
    """ Plots a sample digit from a dataset. """
    X_sample, y_sample = data[random.randrange(len(data))]
    plt.gray()
    plt.imshow(np.reshape(X_sample, (28, 28)))
    plt.title('Label: %d' % (np.argmax(y_sample)))
    plt.show()

def find_failures(model, data):
    """ Finds and plots the digits that the model does not correctly classify. """

    # get model predictions
    predictons = [model.predict(x) for x, _ in data]

    # find misclassified images
    failures = [i for i, (_, y) in enumerate(data) if predictons[i]!=np.argmax(y)]

    for index in failures:

        # plot the image
        plt.gray()
        plt.imshow(np.reshape(data[index][0], (28, 28)))
        plt.title('Predicted label: {0}, actual label: {1}'.format(predictons[index], np.argmax(data[index][1])))
        plt.show()

if __name__ == '__main__':
    main()