from csv import reader
import gzip
import shutil
import os.path
import numpy as np
import math

def load_train(path, ravel=False, percentage=1.0, shuffle=True):
    m = 60000
    u = int(math.floor(percentage*m))
    perm = np.random.permutation(m)[0:u] if shuffle else np.arange(u)

    X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte.gz'), perm, ravel)
    y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'), perm)

    return X_train, y_train

def load_test(path, ravel=False, percentage=1.0, shuffle=True): # first 5000 of test set are easier than last 5000
    m = 10000
    u = int(math.floor(percentage*m))
    perm = np.random.permutation(m)[0:u] if shuffle else np.arange(u)

    X_test = load_images(os.path.join(path, 'train-images-idx3-ubyte.gz'), perm, ravel)
    y_test = load_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'), perm)

    return X_test, y_test

def load_labels(filename, perm):
    labels = np.empty((len(perm)))
    with gzip.open(filename, 'rb') as f:
        offset = 8
        b = f.read()
        for i in range(len(perm)):
            labels_padded = np.zeros((10))
            k = perm[i]
            y = b[k+offset]
            labels_padded[y] = 1
            labels[i] = labels_padded
    return labels

def load_images(filename, perm, ravel):
    n = 28 # image side length
    m = len(perm) # number of rows in data
    
    with gzip.open(filename, 'rb') as f:
        offset = 16
        b = f.read()
        if ravel:
            images = np.empty((m, n*n))
            for i in range(m):
                k = perm[i]
                for j in range(n*n):
                    p = k*n*n + offset + j
                    images[i, j] = b[p]
            return images
        else:
            images = np.empty((m, n, n))
            for p in range(m):
                k = perm[p]
                for i in range(n):
                    for j in range(n):
                        images[p][i][j] = b[(k*n*n) + (i*n) + j + offset]
            return images