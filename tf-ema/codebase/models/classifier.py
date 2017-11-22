import tensorflow as tf
import tensorbayes as tb
from extra_layers import basic_accuracy
from tensorbayes.layers import placeholder, constant
from codebase.args import args
from pprint import pprint
exec "from designs import {:s} as des".format(args.design)
sigmoid_xent = tf.nn.sigmoid_cross_entropy_with_logits
softmax_xent = tf.nn.softmax_cross_entropy_with_logits
import numpy as np

def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

def classifier():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=config),
        src_x = placeholder((None, 32, 32, 3),  name='source_x'),
        src_y = placeholder((None, 10),         name='source_y'),
        test_x = placeholder((None, 32, 32, 3), name='test_x'),
        test_y = placeholder((None, 10),        name='test_y'),
        phase = placeholder((), tf.bool,        name='phase')
    ))

    # Classification
    src_y = des.classifier(T.src_x, T.phase, internal_update=True)
    loss_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_y))
    src_acc = basic_accuracy(T.src_y, src_y)

    # Evaluation (non-EMA)
    test_y = des.classifier(T.test_x, phase=False, reuse=True)
    test_acc = basic_accuracy(T.test_y, test_y)
    fn_test_acc = tb.function(T.sess, [T.test_x, T.test_y], test_acc)

    # Evaluation (EMA)
    ema = tf.train.ExponentialMovingAverage(decay=0.998)
    var_class = tf.get_collection('trainable_variables', 'class')
    ema_op = ema.apply(var_class)
    ema_y = des.classifier(T.test_x, phase=False, reuse=True, getter=get_getter(ema))
    ema_acc = basic_accuracy(T.test_y, ema_y)
    fn_ema_acc = tb.function(T.sess, [T.test_x, T.test_y], ema_acc)

    # Optimizer
    loss_main = loss_class
    var_main = var_class
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)
    train_main = tf.group(train_main, ema_op)

    # Summarizations
    summary_main = [
        tf.summary.scalar('class/loss_class', loss_class),
        tf.summary.scalar('acc/src_acc', src_acc),
    ]
    summary_main = tf.summary.merge(summary_main)

    # Saved ops
    c = tf.constant
    T.ops_print = [
        c('class'), loss_class,
        c('src'), src_acc,
    ]

    T.ops_main = [summary_main, train_main]
    T.fn_test_acc = fn_test_acc
    T.fn_ema_acc = fn_ema_acc

    return T
