import numpy as np
import os
from scipy.io import loadmat
import scipy
import sys
import cPickle as pkl
import tensorbayes as tb
from itertools import izip

def u2t(x):
    """
    Convert uint-8 encoding to 'tanh' encoding (aka range [-1, 1])
    """
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    """
    Convert 'sigmoid' encoding (aka range [0, 1]) to 'tanh' encoding
    """
    return x * 2 - 1

def create_labeled_data(x, y, seed, npc):
    print "Create labeled data, npc:", npc
    state = np.random.get_state()
    np.random.seed(seed)
    shuffle = np.random.permutation(len(x))
    x, y = x[shuffle], y[shuffle]
    np.random.set_state(state)

    x_l, y_l, i_l = [], [], []
    for k in xrange(10):
        idx = y.argmax(-1) == k
        x_l += [x[idx][:npc]]
        y_l += [y[idx][:npc]]
    x_l = np.concatenate(x_l, axis=0)
    y_l = np.concatenate(y_l, axis=0)
    return x_l, y_l

class Data(object):
    def __init__(self, images, labels=None, labeler=None, cast=False):
        self.images = images
        self.labels = labels
        self.labeler = labeler
        self.cast = cast

    def preprocess(self, x):
        if self.cast:
            return u2t(x)
        else:
            return x

    def next_batch(self, bs):
        idx = np.random.choice(len(self.images), bs, replace=False)
        x = self.preprocess(self.images[idx])
        y = self.labeler(x) if self.labels is None else self.labels[idx]
        return x, y

class Cifar10(object):
    def __init__(self, seed=0, npc=None, path='data'):
        print "Loading CIFAR10"
        sys.stdout.flush()
        train = loadmat(os.path.join(path, 'cifar10_train.mat'))
        test = loadmat(os.path.join(path, 'cifar10_test.mat'))

        # Get data
        trainx, trainy = train['X'], train['y'].reshape(-1)
        testx, testy = test['X'], test['y'].reshape(-1)

        # Convert to one-hot
        trainy = np.eye(10)[trainy]
        testy = np.eye(10)[testy]

        # Filter via npc if not None
        if npc:
            trainx, trainy = create_labeled_data(trainx, trainy, seed, npc)

        # Cast is true since data is uint-8 encoded
        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)
