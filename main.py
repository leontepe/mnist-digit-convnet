import matplotlib.pyplot as plt
import numpy as np
import random
import mnist
import neural_network
import sys
import time

def main():

    print('Loading data...')
    start = time.time()
    training_data, test_data = mnist.load_data()
    end = time.time()
    print('Time elapsed: %.2fs' % (round(end-start, 2)))
    
    print('Initializing model...')
    model = neural_network.NeuralNetwork([784, 100, 30, 10])
    
    print('Training model...')
    start = time.time()
    model.stochastic_gradient_descent(training_data, epochs=20, mini_batch_size=20, learning_rate=1e-1, test_data=test_data)
    end = time.time()
    print('Time elapsed: %.2fs' % (round(end-start, 2)))

    print('Plotting misclassified images...')
    find_failures(model, test_data)

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