import tensorflow as tf
import tensorbayes as tb
from codebase.args import args
from utils import delete_existing, save_accuracy as save_acc
import os
import sys
import numpy as np

def update_dict(M, feed_dict, src=None, trg=None, add_z=False, bs=100, y_prior=None):
    if src:
        src_x, src_y = src.train.next_batch(bs)
        feed_dict.update({M.src_x: src_x, M.src_y: src_y})

    if trg:
        trg_x, trg_y = trg.train.next_batch(bs)
        feed_dict.update({M.trg_x: trg_x, M.trg_y: trg_y})

    if add_z:
        y = np.random.multinomial(1, y_prior, size=bs)
        z = np.random.randn(bs, 100)
        feed_dict.update({M.fake_y: y, M.fake_z: z})

def train(M, src=None, trg=None, has_disc=True, add_z=False,
          saver=None, model_name=None, y_prior=None):
    log_dir = os.path.join(args.logdir, model_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)

    if saver:
        model_dir = os.path.join('checkpoints', model_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)

    bs = 64
    iterep = 1000
    n_epoch = 200
    epoch = 0
    feed_dict = {M.phase: 1}

    if src: print "Src size:", src.train.images.shape
    if trg: print "Trg size:", trg.train.images.shape
    print "Batch size:", bs
    print "Iterep:", iterep
    print "Total iterations:", n_epoch * iterep
    print "Log directory:", log_dir

    for i in xrange(n_epoch * iterep):
        # Discriminator
        if has_disc:
            update_dict(M, feed_dict, src, trg, add_z, bs, y_prior)
            summary, _ = M.sess.run(M.ops_disc, feed_dict)
            train_writer.add_summary(summary, i + 1)

        # Main
        update_dict(M, feed_dict, src, trg, add_z, bs, y_prior)
        summary, _ = M.sess.run(M.ops_main, feed_dict)
        train_writer.add_summary(summary, i + 1)

        train_writer.flush()
        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(epoch, i),
                                            display=args.run >= 999)

        if end_epoch:
            print_list = M.sess.run(M.ops_print, feed_dict)

            if src:
                save_acc(M, 'fn_test_acc', 'test/src_test',
                         src.test,  train_writer, i, print_list)
                save_acc(M, 'fn_ema_acc', 'test/src_ema',
                         src.test,  train_writer, i, print_list)

            print_list += ['epoch', epoch]
            print print_list
            sys.stdout.flush()

        if end_epoch and epoch % 20 == 0:
            if hasattr(M, 'ops_image'):
                summary = M.sess.run(M.ops_image, feed_dict)
                train_writer.add_summary(summary, i + 1)

        if saver and (i + 1) % 20000 == 0:
            path = saver.save(M.sess,
                              os.path.join(model_dir, 'model'),
                              global_step=i + 1)
            print "Saving model to {:s}".format(path)
            sys.stdout.flush()

    # Saving final model just in case
    if saver:
        path = saver.save(M.sess,
                          os.path.join(model_dir, 'model'),
                          global_step=i + 1)
        print "Saving model to {:s}".format(path)
        sys.stdout.flush()
