import subprocess
from scipy.io import loadmat, savemat
import numpy as np
import os

def get_data_from_mat(fs):
    Xs, ys = [], []
    for f in fs:
        print "Opening {}".format(f)
        mat = loadmat(f)
        X = mat['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y = mat['labels'].reshape(-1)
        Xs += [X]
        ys += [y]
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    data = {'X': X, 'y': y}
    return data

def save_data_to_mat(f, data):
    print "Saving {}".format(f)
    savemat(f, data)

def main():
    if os.path.exists('raw_data'):
        print "Using existing raw_data directory"
    else:
        print "Opening subprocess to download data from URL"
        subprocess.check_output(
            '''
            wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz -P raw_data
            cd raw_data
            tar -xzvf cifar-10-matlab.tar.gz
            rm cifar-10-matlab.tar.gz
            ''',
            shell=True)

    if not os.path.exists('data'):
        os.makedirs('data')

    base = 'raw_data/cifar-10-batches-mat'
    data = get_data_from_mat(['{}/data_batch_{}'.format(base, i) for i in xrange(1, 6)])
    save_data_to_mat('data/cifar10_train.mat', data)
    data = get_data_from_mat(['{}/test_batch.mat'.format(base)])
    save_data_to_mat('data/cifar10_test.mat', data)

if __name__ == '__main__':
    main()
