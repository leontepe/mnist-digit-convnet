import matplotlib.pyplot as plt
import numpy as np
import random
import data
import neural_network
import sys
import time

def main():

    print('Loading training data...')
    start = time.time()
    training_data = data.load_train('data')
    end = time.time()
    print('Time elapsed: %.2fs' % (round(end-start, 2)))
    
    print('Loading test data...')
    start = time.time()
    test_data = data.load_test('data')
    end = time.time()
    print('Time elapsed: %.2fs' % (round(end-start, 2)))
    
    print('Initializing model...')
    model = neural_network.NeuralNetwork((28*28, 30, 10))
    
    print('Training model...')
    start = time.time()
    model.stochastic_gradient_descent(training_data, test_data, 5, 10, 3)
    end = time.time()
    print('Time elapsed: %.2fs' % (round(end-start, 2)))

def sample(data):
    X_sample, y_sample = data[random.randrange(len(data))]
    plt.gray()
    plt.imshow(np.reshape(X_sample, (28, 28)))
    plt.title('Label: %d' % (np.argmax(y_sample)))
    plt.show()

if __name__ == '__main__':
    main()