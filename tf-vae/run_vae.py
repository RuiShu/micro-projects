import os
import sys
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf

DATA = '/home/ruishu/data/'

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--trg',       type=str,   default='svhn',  help="Trg data")
parser.add_argument('--nn',        type=str,   default='v1',    help="Architecture")
parser.add_argument('--Z',         type=int,   default=10,      help="Z dimensionality")
parser.add_argument('--lr',        type=float, default=1e-3,    help="Learning rate")
parser.add_argument('--run',       type=int,   default=999,     help="Run index")
parser.add_argument('--datadir',   type=str,   default=DATA,    help="Data directory")
parser.add_argument('--logdir',    type=str,   default='log',   help="Log directory")
codebase_args.args = args = parser.parse_args()
pprint(vars(args))

from codebase.datasets import get_data
from codebase.models.vae import vae
from codebase.train import train

# Make model name
setup = [
    ('model={:s}',  'vae'),
    ('trg={:s}',  args.trg),
    ('nn={:s}', args.nn),
    ('Z={:d}', args.Z),
    ('lr={:.0e}',  args.lr),
    ('run={:04d}',   args.run)
]

model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

M = vae()
M.sess.run(tf.global_variables_initializer())
src = None
trg = get_data(args.trg)
saver = None # tf.train.Saver()

train(M, src, trg,
      saver=saver,
      model_name=model_name)
