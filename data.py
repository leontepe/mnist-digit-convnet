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
    return X_train, y_train

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
    images = list()
    x = 0
    with gzip.open(filename, 'rb') as f:
        b = f.read()
        offset = 16
        n = 28
        for k in perm:
            image = list()
            for i in range(n):
                row = list()
                for j in range(n):
                    val = b[(k*n*n) + (i*n) + j + offset]
                    row.append(val)
                image.append(row)
            images.append(image)
    return images