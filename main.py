import random
import data
from classifiers import ConvolutionalNeuralNetwork as CNN
import matplotlib.pyplot as plt
import numpy as np

def main():
    random.seed(42)
    X_train, y_train = data.load_train('data', 0.01) # reduce dataset size for debugging purposes
    display_image(X_train[random.randrange(len(X_train))])

def display_image(data):
    #data = np.array(data)
    plt.gray()
    plt.imshow(data)
    plt.show()

if __name__ == '__main__':
    main()