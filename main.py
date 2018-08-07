import random
import data
from classifiers import NeuralNetwork as NN
import matplotlib.pyplot as plt
import numpy as np

def main():
    #random.seed(42)
    #X_train, y_train = data.load_train('data', ravel=True, percentage=0.01, shuffle=True)
    #sample(X_train, y_train, unravel=True)

    a = np.array([1, 2, 3, 2, 2, 2, 4])
    b = np.array([0, 1, 4, 3, 2, 0, 1])

    print(c)

def sample(X, y, unravel):
    index = random.randrange(len(X))
    data = (X[index, :]).reshape((28, 28)) if unravel else X[index]
    plt.gray()
    plt.imshow(data)
    plt.title('Label: %d' % (np.argmax(y[index])))
    plt.show()

if __name__ == '__main__':
    main()