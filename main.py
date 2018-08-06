import random
import data
from classifiers import ConvolutionalNeuralNetwork as CNN
import matplotlib.pyplot as plt
import numpy as np

def main():
    random.seed(42)
    X_train, y_train = data.load_train('data', 0.01) # reduce dataset size for debugging purposes
    sample(X_train, y_train)

def sample(X, y):
    index = random.randrange(len(X))
    plt.gray()
    plt.imshow(X[index])
    plt.title('Label: %d' % (y[index]))
    plt.show()

if __name__ == '__main__':
    main()