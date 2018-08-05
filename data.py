from csv import reader
import gzip
import shutil
import os.path
import numpy as np
import math

def load_train(path, percentage=1.0):
    m = 60000
    perm = np.random.permutation(m)[0:int(math.floor(percentage*m))]
    X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte.gz'), perm)
    y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'), perm)
    return np.array(X_train), np.array(y_train)

def load_test(path, percentage=1.0): # first 5000 of test set are easier than last 5000
    m = 10000
    perm = np.random.permutation(m)[0:int(math.floor(percentage*m))]
    X_test = load_images(os.path.join(path, 't10k-images-idx3-ubyte.gz'), perm)
    y_test = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte.gz'), perm)
    return X_test, y_test

def load_labels(filename, perm):
    labels = list()
    with gzip.open(filename, 'rb') as f:
        b = f.read()
        offset = 8
        for i in perm + offset:
            labels.append(b[i])
    return labels

def load_images(filename, perm):
    len_side = 28
    images = np.empty((len(perm), len_side, len_side))
    with gzip.open(filename, 'rb') as f:
        b = f.read()
        offset = 16
        for p in range(len(perm)):
            k = perm[p]
            for i in range(len_side):
                for j in range(len_side):
                    images[p][i][j] = b[(k*len_side**2) + (i*len_side) + j + offset]
    return images