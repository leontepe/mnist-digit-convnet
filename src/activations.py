import numpy as np

def sigmoid(z, derivative=False):
    """ The sigmoid activation function and its derivative. """ 
    if not derivative:
        return 1.0/(1.0 + np.exp(-z))
    else:
        return np.multiply(sigmoid(z), sigmoid(1-z))

def get(identifier):
    glb = globals()
    if identifier in glb:
        val = glb[identifier]
        if callable(val):
            return val
        else:
            raise ValueError('Activation function identifier does not correspond to a function:', identifier)
    else:
        raise ValueError('Could not interpret activation function identifier:', identifier)