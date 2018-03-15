import os
import numpy as np
import scipy
import sys
import tensorbayes as tb
from codebase.args import args
from scipy.io import loadmat
from itertools import izip
from utils import u2t

def get_info(domain_id, domain):
    train, test = domain.train, domain.test
    print '{} info'.format(domain_id)
    print 'Train X/Y shapes: {}, {}'.format(train.images.shape, train.labels.shape)
    print 'Train X min/max/cast: {}, {}, {}'.format(
        train.images.min(),
        train.images.max(),
        train.cast)
    print 'Test shapes: {}, {}'.format(test.images.shape, test.labels.shape)
    print 'Test X min/max/cast: {}, {}, {}\n'.format(
        test.images.min(),
        test.images.max(),
        test.cast)

class Data(object):
    def __init__(self, images, labels=None, cast=False):
        """Data object constructs mini-batches to be fed during training

        images - (NHWC) data
        labels - (NK) one-hot data
        labeler - (tb.function) returns simplex value given an image
        cast - (bool) converts uint8 to [-1, 1] float
        """
        self.images = images
        self.labels = labels
        self.cast = cast

    def preprocess(self, x):
        if self.cast:
            return u2t(x)
        else:
            return x

    def next_batch(self, bs):
        """Constructs a mini-batch of size bs without replacement
        """
        idx = np.random.choice(len(self.images), bs, replace=False)
        x = self.preprocess(self.images[idx])
        y = self.labels[idx]
        return x, y

class Svhn(object):
    def __init__(self, train='train'):
        """SVHN domain train/test data
        train - (str) flag for using 'train' or 'extra' data
        """
        print "Loading SVHN"
        train = loadmat(os.path.join(args.datadir, '{:s}_32x32.mat'.format(train)))
        test = loadmat(os.path.join(args.datadir, 'test_32x32.mat'))

        # Change format
        trainx, trainy = self.change_format(train)
        testx, testy = self.change_format(test)

        self.train = Data(trainx, trainy, cast=True)
        self.test = Data(testx, testy, cast=True)

    @staticmethod
    def change_format(mat):
        """Convert X: (HWCN) -> (NHWC) and Y: [1,...,10] -> one-hot
        """
        x = mat['X'].transpose((3, 0, 1, 2))
        y = mat['y'].reshape(-1)
        y[y == 10] = 0
        y = np.eye(10)[y]
        return x, y

def get_data(domain_id):
    """Returns Domain object based on domain_id
    """
    if domain_id == 'svhn':
        return Svhn(train='extra')

    else:
        raise Exception('dataset {:s} not recognized'.format(domain_id))
