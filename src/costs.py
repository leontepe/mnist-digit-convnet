import numpy as np
import activations

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * activations.sigmoid(z, derivative=True)

def categorical_cross_entropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def mean_squared_error(a, y):
    return (1/2) * np.linalg.norm(a - y) ** 2

def get(identifier):
    if callable(identifier):
        return getattr(identifier)
    else:
        raise ValueError('Could not interpret loss function identifier:', identifier)