import matplotlib.pyplot as plt
import numpy as np
import random
import mnist
from neural_network import NeuralNetwork
import sys
import time
import os.path
import activations

def main():
    """ Main entry point of the script. """

    """
    plot_failures = False
    filename = 'network.json'

    print('Loading data...')
    start = time.time()
    training_data, test_data = mnist.load_data()
    end = time.time()
    print('Time elapsed: %.2fs' % (round(end-start, 2)))
    
    if os.path.exists(filename):
        print('Found an existing model. Deserializing...', end=' ')
        model = NeuralNetwork.deserialize(filename)
        print('Finished')
    else:
        print('Initializing new model...', end=' ')
        model = NeuralNetwork([784, 100, 10])
        print('Finished')
    
    print('Training model...')
    start = time.time()
    model.mini_batch_gradient_descent(training_data, epochs=3, mini_batch_size=10, learning_rate=3, regularization_parameter=1, test_data=test_data)
    end = time.time()
    print('Time elapsed: %.2fs' % (round(end-start, 2)))

    print('Serializing model...', end=' ')
    model.serialize(filename)
    print('Finished')

    if plot_failures:
        print('Plotting misclassified images...')
        find_failures(model, test_data)
    """

    sigmoid = activations.get('sigmoid')
    print(sigmoid(0))

    shape1 = (283,)
    shape2 = (49,)
    merged = shape1, shape2
    print(merged[0])

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