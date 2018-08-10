from csv import reader
import gzip
import shutil
import os.path
import numpy as np
import math

def load_train(path):
    X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte.gz'))
    y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'))
    return merge(X_train, y_train)

def load_test(path):
    X_train = load_images(os.path.join(path, 't10k-images-idx3-ubyte.gz'))
    y_train = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte.gz'))
    return merge(X_train, y_train)
    
def load_images(filename):
    images = []
    offset = 16
    num_pixels = 28*28
    with gzip.open(filename, 'r') as f:
        content = f.read()
        num_rows = int((len(content) - offset) / num_pixels)
        #printProgressBar(0, num_rows, 'Loading images:')
        for i in range(num_rows):
            # index of first pixel
            k = i*num_pixels + offset
            # create n by 1 vector of pixels
            img = np.reshape(list(content[k:k+num_pixels]), (num_pixels, 1))
            # append image to image list
            images.append(img)
            #printProgressBar(i, num_rows, 'Loading images:')
    #print()
    return images

def load_labels(filename):
    labels = []
    offset = 8
    with gzip.open(filename, 'r') as f:
        content = f.read()
        num_rows = len(content) - offset
        #printProgressBar(0, num_rows, 'Loading labels:')
        for i in range(num_rows):
            y = content[offset+i]
            labels.append(np.array([i==y for i in range(10)]).reshape((10, 1)))
            #printProgressBar(i, num_rows, 'Loading labels:')
    #print()
    return labels

def merge(X, y):
    return [(X, y) for X, y in zip(X, y)]

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()