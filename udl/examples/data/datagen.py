__author__ = 'haitham'

import sklearn
import sklearn.datasets
import sklearn.linear_model
import os
import numpy as np
import cPickle
import struct
from array import array

def get_sklearn_data():
    X, Y = sklearn.datasets.make_classification(
        n_samples=10000, n_features=4, n_redundant=0, n_informative=2,
        n_clusters_per_class=2, hypercube=False, random_state=0
    )

    # Split into train and test
    x_train, y_train, x_test, y_test = sklearn.cross_validation.train_test_split(X, Y)
    return x_train, y_train, x_test, y_test


def load_cifar_batch(file):
    
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_cifar_data(directory):
	trainign_batch = 'data_batch_1'
	testing_batch = 'test_batch'
	batch = os.path.join(directory, trainign_batch)
	dict_train = load_cifar_batch(batch)
	batch = os.path.join(directory, testing_batch)
	dict_test = load_cifar_batch(batch)	
	
	x_train = dict_train['data']
	y_train = np.asarray(dict_train['labels'])
	x_test = dict_test['data']
	y_test = np.asarray(dict_test['labels'])
	return x_train, y_train, x_test, y_test

def load_mnist_file(x_file, y_file):
    labels_file = open(y_file, 'rb')
    magic_nr, size = struct.unpack(">II", labels_file.read(8))
    y = array("b", labels_file.read())
    y=np.array(y)
    labels_file.close()

    images_file = open(x_file, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", images_file.read(16))
    images = array("B", images_file.read())
    images_file.close()

    x = []
    for i in xrange(size):
        x.append([0]*rows*cols)

    for i in xrange(size):
        x[i][:] = images[i*rows*cols : (i+1)*rows*cols]
        #x[i, :] = images[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
    x  = np.array(x)
    y= np.array(y)
    
    return x, y


def get_mnist_data(directory):

    x_train_file = os.path.join(directory, 'train-images.idx3-ubyte')
    y_train_file = os.path.join(directory, 'train-labels.idx1-ubyte')
   
    x_test_file = os.path.join(directory, 't10k-images.idx3-ubyte')
    y_test_file = os.path.join(directory, 't10k-labels.idx1-ubyte')
   
    x_train,y_train = load_mnist_file(x_train_file,y_train_file)
    x_test,y_test = load_mnist_file(x_test_file,y_test_file)


    #ind = [ k for k in xrange(size) if lbl[k] in digits ]
    #images =  matrix(0, (len(ind), rows*cols))
    #labels = matrix(0, (len(ind), 1))
    #for i in xrange(len(ind)):
     #   images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
      #  labels[i] = lbl[ind[i]]

    return x_train,y_train, x_test,y_test
