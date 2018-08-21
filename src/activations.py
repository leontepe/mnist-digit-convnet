import numpy as np

def sigmoid(z, derivative=False):
    """ The sigmoid activation function and its derivative. """ 
    if not derivative:
        return 1.0/(1.0 + np.exp(-z))
    else:
        return np.multiply(sigmoid(z), sigmoid(1-z))