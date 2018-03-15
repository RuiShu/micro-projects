import tensorflow as tf
import tensorbayes as tb
import numpy as np
from codebase.args import args
from codebase.utils import t2s
from tensorbayes.distributions import log_normal
from tensorbayes.layers import constant
from tensorbayes.tfutils import reduce_l2_loss
from tensorflow.contrib.framework import add_arg_scope

@add_arg_scope
def leaky_relu(x, a=0.2, scope=None):
    with tf.name_scope(scope, 'leaky_relu'):
        return tf.maximum(x, a * x)

@add_arg_scope
def vae_loss(x, x_prior, z, z_post, scope=None):
    with tf.name_scope(scope, 'vae_loss'):
        z_prior = (0., 1.)
        loss_kl = tf.reduce_mean(log_normal(z, *z_post) - log_normal(z, *z_prior))
        loss_rec = tf.reduce_mean(reduce_l2_loss(x - x_prior, axis=[1,2,3]))
        loss_gen = loss_kl + loss_rec

    return loss_rec, loss_kl, loss_gen

@add_arg_scope
def generate_image(generator):
    ncol = 20
    with tb.nputils.FixedSeed(0):
        z = np.random.randn(10 * ncol, args.Z)

    z = constant(z)
    img = generator(z, phase=False, reuse=True)
    img = tf.reshape(img, [10, ncol, 32, 32, 3])
    img = tf.reshape(tf.transpose(img, [0, 2, 1, 3, 4]), [1, 10 * 32, ncol * 32, 3])
    return t2s(img)
