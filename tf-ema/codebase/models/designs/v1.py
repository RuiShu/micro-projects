from codebase.models.extra_layers import leaky_relu
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorbayes.layers import dense, conv2d, avg_pool, max_pool, batch_norm
from codebase.args import args

dropout = tf.layers.dropout

def classifier(x, phase, scope='class', reuse=None, internal_update=False, getter=None):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            layout = [
                (conv2d, (96, 3, 1), {}),
                (conv2d, (96, 3, 1), {}),
                (conv2d, (96, 3, 1), {}),
                (max_pool, (2, 2), {}),
                (dropout, (), dict(training=phase)),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (max_pool, (2, 2), {}),
                (dropout, (), dict(training=phase)),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (avg_pool, (), dict(global_pool=True)),
                (dense, (10,), dict(activation=None))
            ]

            start = 0
            end = len(layout)

            for i in xrange(start, end):
                with tf.variable_scope('l{:d}'.format(i)):
                    f, f_args, f_kwargs = layout[i]
                    x = f(x, *f_args, **f_kwargs)

    return x
