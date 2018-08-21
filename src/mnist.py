from csv import reader
import gzip
import shutil
import os.path
import numpy as np
import math

def load_data():
    return load_train(), load_test()

def load_train():
    """ Returns the MNIST training dataset as a `list` of `(x, y)` tuples. """
    X_train = load_images('data/train-images-idx3-ubyte.gz')
    y_train = load_labels('data/train-labels-idx1-ubyte.gz')
    return merge(X_train, y_train)

def load_test():
    """ Returns the MNIST test dataset as a `list` of `(x, y)` tuples. """
    X_train = load_images('data/t10k-images-idx3-ubyte.gz')
    y_train = load_labels('data/t10k-labels-idx1-ubyte.gz')
    return merge(X_train, y_train)
    
def load_images(filename, do_normalization=True):
    """
    Returns a `list` of `numpy.ndarray`s of shape `(784,1)` representing the image. Unless specified otherwise, the values will be normalized.
    """
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
    return normalize(images) if do_normalization else images

def load_labels(filename):
    """ Returns a `list` of labels where each one is a `numpy.ndarray` of shape `(10,1)` representing the desired network output. """
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

def normalize(x):
    """ Returns a normalized version of the input image. """
    return [x/255 for x in x]

def merge(x, y):
    """ Returns a merged `list` of `(x, y)` tuples from seperate `x` and `y` lists. """
    return [(x, y) for x, y in zip(x, y)]

# copied this from some stackoverflow thread
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