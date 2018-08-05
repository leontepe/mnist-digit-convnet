import random
import data
from classifiers import ConvolutionalNeuralNetwork as CNN

def main():
    random.seed(42)
    X_train, y_train = data.load_train('data', 0.01) # takes ~20 seconds to load
    #display_image(X_train[random.randrange(len(X_train))])
    print(len(X_train))

def display_image(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 0:
                print('#', end='')
            else:
                print('O', end='')

if __name__ == '__main__':
    main()