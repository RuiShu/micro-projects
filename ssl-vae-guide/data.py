import tensorbayes as tb
from tensorbayes.nputils import convert_to_ssl
import numpy as np
import pickle as pkl
import os, urllib, gzip
import scipy.io

class Mnist(object):
    def __init__(self, n_labels, seed, binarize=True, duplicate=True):
        self._load_mnist()
        if binarize: 
            self.binarize()
        self.convert_to_ssl(n_labels, seed, duplicate)

    def next_batch(self, bs):
        xu_idx = np.random.choice(len(self.x_train), bs, replace=False)
        yu_idx = np.random.choice(len(self.y_train), bs, replace=False)
        l_idx = np.random.choice(len(self.x_label), bs, replace=False)
        return self.x_label[l_idx], self.y_label[l_idx], self.x_train[xu_idx], self.y_train[yu_idx]

    @staticmethod
    def _download_mnist():
        folder = os.path.join('data', 'mnist_real')
        data_loc = os.path.join(folder, 'mnist.pkl.gz')
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(data_loc):
            url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print "Downloading data from:", url
            urllib.urlretrieve(url, data_loc)
        return data_loc

    def _load_mnist(self):
        f = gzip.open(self._download_mnist(), 'rb')
        train, valid, test = pkl.load(f)
        f.close()
        self.x_train, self.y_train = train[0], train[1]
        self.x_valid, self.y_valid = valid[0], valid[1]
        self.x_test, self.y_test = test[0], test[1]

    def binarize(self, seed=42):
        state = np.random.get_state()
        np.random.seed(seed)
        self.x_train = np.random.binomial(1, self.x_train)
        self.x_valid = np.random.binomial(1, self.x_valid)
        self.x_test  = np.random.binomial(1, self.x_test)
        np.random.set_state(state)

    def convert_to_ssl(self, n_labels, seed, duplicate):
        state = np.random.get_state()
        np.random.seed(seed)
        if n_labels == 50000:
            # Be very careful: if x_label and x_train are binarized
            # differently, we actually accidentally increase our dataset size
            print "Using full data set. No conversion used"
            self.x_label, self.y_label = np.copy(self.x_train), np.copy(self.y_train)
        else:
            xl, yl, xu, yu = tb.nputils.convert_to_ssl(self.x_train,
                                                       self.y_train,
                                                       n_labels,
                                                       n_classes=10,
                                                       complement=not duplicate)
            self.x_label, self.y_label = xl, yl
            self.x_train, self.y_train = xu, yu
        np.random.set_state(state)
