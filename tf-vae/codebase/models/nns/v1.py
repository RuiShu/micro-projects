import numpy as np
import tensorflow as tf
from codebase.args import args
from codebase.models.extra_layers import leaky_relu
from tensorbayes.layers import dense, conv2d, conv2d_transpose, batch_norm, gaussian_sample
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.layers.core import dropout

def encoder(x, y=None, phase=False, scope='enc', reuse=None, internal_update=False):
    with tf.variable_scope(scope, reuse=reuse):
        with arg_scope([conv2d, dense], bn=True, phase=phase, activation=leaky_relu), \
             arg_scope([batch_norm], internal_update=internal_update):

            x = conv2d(x, 32, 3, 2)
            x = conv2d(x, 64, 3, 2)
            x = conv2d(x, 128, 3, 2)
            x = dense(x, 1024)

            m = dense(x, args.Z, activation=None)
            v = dense(x, args.Z, activation=tf.nn.softplus) + 1e-5
            z = gaussian_sample(m, v)

    return z, (m, v)

def generator(z, y=None, phase=False, scope='gen', reuse=None, internal_update=False):
    with tf.variable_scope(scope, reuse=reuse):
        with arg_scope([dense, conv2d_transpose], bn=True, phase=phase, activation=leaky_relu), \
             arg_scope([batch_norm], internal_update=internal_update):

            if y is not None:
                z = tf.concat([z, y], 1)

            z = dense(z, 4 * 4 * 512)
            z = tf.reshape(z, [-1, 4, 4, 512])
            z = conv2d_transpose(z, 128, 5, 2)
            z = conv2d_transpose(z, 64, 5, 2)
            x = conv2d_transpose(z, 3, 5, 2, bn=False, activation=tf.nn.tanh)

    return x
