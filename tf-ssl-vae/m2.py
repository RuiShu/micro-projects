import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-run", type=int, help="Run index. Use 0 if first run.", required=True)
parser.add_argument("-n_labels", type=int, help="Number of labeled data.", default=100)
parser.add_argument('-seed', type=int, help='Seed for semi-sup conversion', default=0)
parser.add_argument("-nonlin", type=str, help="Activation function.", default='elu')
parser.add_argument("-eps", type=float, help="Distribution epsilon.", default=1e-5)
parser.add_argument("-bs", type=int, help="Minibatch size.", default=100)
parser.add_argument("-lr", type=float, help="Learning rate.", default=5e-4)
args = parser.parse_args()

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
import tensorbayes as tb
from tensorbayes.layers import *
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal
from data import Mnist

log_file = 'results/n_labels={:d}/m2_run={:d}.csv'.format(args.n_labels, args.run)
writer = tb.FileWriter(log_file, args=args, pipe_to_sys=True, overwrite=args.run >= 999)
if args.nonlin == 'relu':
    activate = tf.nn.relu
elif args.nonlin == 'elu':
    activate = tf.nn.elu
else:
    raise Exception("Unexpected nonlinearity arg")
log_bern = lambda x, logits: log_bernoulli_with_logits(x, logits, args.eps)
log_norm = lambda x, mu, var: log_normal(x, mu, var, 0.0)

def name(idx, suffix):
    return 'L{:d}'.format(idx) + '_' + suffix

def encode_block(x, h_size, z_size, idx, depth=2, reuse=False, discrete=False):
    with tf.variable_scope(name(idx, 'encode'), reuse=reuse):
        h = x
        for d in xrange(depth):
            h = dense(h, h_size, 'layer{:d}'.format(d), activation=activate)
    if not discrete:
        with tf.variable_scope(name(idx, 'encode/likelihood'), reuse=reuse):
            z_m = dense(h, z_size, 'mean')
            z_v = dense(h, z_size, 'var', activation=tf.nn.softplus) + args.eps
        return (z_m, z_v)
    else:
        with tf.variable_scope(name(idx, 'encode/discrete'), reuse=reuse):
            logits = dense(h, z_size, 'logits')
        return logits

def infer_block(likelihood, prior, idx, reuse=False):
    with tf.variable_scope(name(idx, 'sample'), reuse=reuse):
        if prior is None:
            posterior = likelihood
        else:
            args = likelihood + prior
            posterior = gaussian_update(*args, scope='pwm')
        z = gaussian_sample(*posterior, scope='sample')
    return z, posterior

def decode_block(z, h_size, x_size, idx, depth=2, reuse=False):
    with tf.variable_scope(name(idx, 'decode'), reuse=reuse):
        h = z
        for d in xrange(depth):
            h = dense(h, h_size, 'layer{:d}'.format(d), activation=activate)
    with tf.variable_scope(name(idx, 'decode/prior'), reuse=reuse):
        if idx == 0:
            logits = dense(h, x_size, 'logits', bn=False)
            return logits
        else:
            x_m = dense(h, x_size, 'mean')
            x_v = dense(h, x_size, 'var', activation=tf.nn.softplus) + args.eps
            x_prior = (x_m, x_v)
            return x_prior

def infer_decode_block(z_like, z_prior, h_size, x_size, idx, depth=2, reuse=False):
    z, z_post = infer_block(z_like, z_prior, idx[0], depth=depth, reuse=reuse)
    x_prior = decode_block(z, h_size, x_size, idx[1], depth=depth, reuse=reuse)
    return z, z_post, x_prior

tf.reset_default_graph()
with tf.name_scope('prior'):
    z_prior = (constant(0), constant(1))
with tf.name_scope('l'):
    x = placeholder((None, 784), name='x')
    y = placeholder((None, 10), name='y')
    with tf.name_scope('preprocess'):
        x = bernoulli_sample(x)
    with tf.variable_scope('class'):
        y_logits = encode_block(x, 500, 10, idx=1, discrete=True)
    # Encode (x, y) and decode (x)
    with tf.variable_scope('vae'):
        z_like = encode_block(tf.concat(1, [x, y]), 500, 50, idx=1)
        z, z_post = infer_block(z_like, None, idx=1)
        x_logits = decode_block(tf.concat(1, [z, y]), 500, 784, idx=0)
    # Compute losses
    with tf.name_scope('loss'):
        rec_x = -log_bern(x, x_logits)
        rec_y = -np.log(0.1)
        kl_z  = -log_norm(z, *z_prior) + log_norm(z, *z_post)
        l_loss = tf.reduce_mean(rec_x + rec_y + kl_z)
    with tf.name_scope('accuracy'):
        acc = tf.reduce_mean(tf.cast(tf.equal(
                    tf.argmax(y, axis=1), tf.argmax(y_logits, axis=1)), 'float32'))
    with tf.name_scope('alpha'):
        a_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_logits, y))

reuse = True
with tf.name_scope('u'):
    x = placeholder((None, 784), name='x')
    with tf.name_scope('preprocess'):
        xb = bernoulli_sample(x)
        y = tf.reshape(tf.tile(constant(np.eye(10).reshape(10, 1, 10)), (1, tf.shape(x)[0], 1)), (-1, 10), name='y')
        x = tf.tile(xb, (10, 1))
    with tf.variable_scope('class', reuse=reuse):
        y_logits = encode_block(xb, 500, 10, idx=1, discrete=True)
    # Encode (x, y) and decode (x)
    with tf.variable_scope('vae', reuse=reuse):
        z_like = encode_block(tf.concat(1, [x, y]), 500, 50, idx=1)
        z, z_post = infer_block(z_like, None, idx=1)
        x_logits = decode_block(tf.concat(1, [z, y]), 500, 784, idx=0)
    # Compute losses
    with tf.name_scope('loss'):
        rec_x = -log_bern(x, x_logits)
        rec_y = -np.log(0.1)
        kl_z  = -log_norm(z, *z_prior) + log_norm(z, *z_post)
        u_loss = tf.transpose(tf.reshape(rec_x + rec_y + kl_z, (10, -1)))
        qy = tf.nn.softmax(y_logits)
        ln_qy = tf.nn.log_softmax(y_logits)
        u_loss = tf.reduce_mean(tf.reduce_sum(u_loss * qy + qy * ln_qy, axis=-1))

wl = placeholder(None, name='wl')
wu = placeholder(None, name='wu')
wa = placeholder(None, name='wa')
with tf.name_scope('loss'):
    y = placeholder((None, 10), name='y')
    loss = wl * l_loss + wu * u_loss + wa * a_loss

train_step = tf.train.AdamOptimizer().minimize(loss)
mnist = Mnist(100, 0, binarize=False, duplicate=False)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

wu = 1.0
wl = 1.0
wa = 1.0
writer.add_var('train_acc', '{:8.3f}', acc)
writer.add_var('train_a_loss', '{:8.3f}', a_loss)
writer.add_var('train_l_loss', '{:8.3f}', l_loss)
writer.add_var('train_u_loss', '{:8.3f}', u_loss)
writer.add_var('test_acc', '{:8.3f}')
writer.add_var('test_a_loss', '{:8.3f}')
writer.add_var('test_l_loss', '{:8.3f}')
writer.add_var('test_u_loss', '{:8.3f}')
writer.add_var('epoch', '{:>8d}')
# Sanity checks
writer._write('# Sanity check')
writer._write('# Labeled size: {:s}'.format(str(mnist.x_label.shape)))
writer._write('# Unlabeled size: {:s}'.format(str(mnist.x_train.shape)))
writer._write('# (wu, wl, wa): {:s}'.format(str((wu, wl, wa))))
writer.initialize()

iterep = 500
for i in range(iterep * 2000):
    x, y, xu, _ = mnist.next_batch(100)
    sess.run([train_step],
              feed_dict={'l/x:0': x,
                        'l/y:0': np.eye(10)[y],
                        'u/x:0': xu,
                        'wu:0': 1.0,
                        'wl:0': 1.0,
                        'wa:0': 1.0})
    end_epoch, epoch = tb.utils.progbar(i, iterep)
    if end_epoch:
        tr = sess.run(writer.tensors, feed_dict={'l/x:0': mnist.x_label, 
                                                 'l/y:0': np.eye(10)[mnist.y_label],
                                                 'u/x:0': mnist.x_train[np.random.choice(len(mnist.x_train), 10000, replace=False)]})
        te = sess.run(writer.tensors, feed_dict={'l/x:0': mnist.x_test, 
                                                 'l/y:0': np.eye(10)[mnist.y_test],
                                                 'u/x:0': mnist.x_test})
        writer.write(tensor_values=tr, values=te + [epoch])
