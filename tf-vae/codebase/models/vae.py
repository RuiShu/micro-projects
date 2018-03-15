import importlib
import numpy as np
import tensorbayes as tb
import tensorflow as tf
from codebase.args import args
from extra_layers import generate_image, vae_loss
from pprint import pprint
from tensorbayes.layers import placeholder, constant
nn = importlib.import_module('codebase.models.nns.{}'.format(args.nn))

def vae():
    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=tb.growth_config()),
        trg_x = placeholder((None, 32, 32, 3),   name='target_x'),
        fake_z = placeholder((None, args.Z), name='fake_z'),
    ))

    # Inference
    z, z_post = nn.encoder(T.trg_x, phase=True, internal_update=True)

    # Generation
    x = nn.generator(z, phase=True, internal_update=True)

    # Loss
    loss_rec, loss_kl, loss_gen = vae_loss(x, T.trg_x, z, z_post)

    # Evaluation (embedding, reconstruction, loss)
    test_z, test_z_post = nn.encoder(T.trg_x, phase=False, reuse=True)
    test_x = nn.generator(test_z, phase=False, reuse=True)
    _, _, test_loss = vae_loss(test_x, T.trg_x, test_z, test_z_post)
    fn_embed = tb.function(T.sess, [T.trg_x], test_z_post)
    fn_recon = tb.function(T.sess, [T.trg_x], test_x)
    fn_loss = tb.function(T.sess, [T.trg_x], test_loss)

    # Evaluation (generation)
    fake_x = nn.generator(T.fake_z, phase=False, reuse=True)
    fn_generate = tb.function(T.sess, [T.fake_z], fake_x)

    # Optimizer
    var_main = tf.get_collection('trainable_variables', 'gen/')
    var_main += tf.get_collection('trainable_variables', 'enc/')
    loss_main = loss_gen
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)

    # Summarizations
    summary_main = [
        tf.summary.scalar('gen/loss_gen', loss_gen),
        tf.summary.scalar('gen/loss_kl', loss_kl),
        tf.summary.scalar('gen/loss_rec', loss_rec),
    ]
    summary_image = [tf.summary.image('gen/gen', generate_image(nn.generator))]

    # Merge summaries
    summary_main = tf.summary.merge(summary_main)
    summary_image = tf.summary.merge(summary_image)

    # Saved ops
    c = tf.constant
    T.ops_print = [
        c('gen'), loss_gen,
        c('kl'), loss_kl,
        c('rec'), loss_rec,
    ]

    T.ops_main = [summary_main, train_main]
    T.ops_image = summary_image
    T.fn_embed = fn_embed
    T.fn_recon = fn_recon
    T.fn_loss = fn_loss
    T.fn_generate = fn_generate

    return T
