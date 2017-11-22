import os
import sys
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--src',    type=str,   default='cifar10', help="Src data")
parser.add_argument('--design', type=str,   default='v1',      help="design")
parser.add_argument('--lr',     type=float, default=1e-3,      help="Learning rate")
parser.add_argument('--run',    type=int,   default=999,       help="Run index")
parser.add_argument('--logdir', type=str,   default='log',     help="Log directory")
codebase_args.args = args = parser.parse_args()
pprint(vars(args))

from codebase.models.classifier import classifier
from codebase.train import train
from codebase.utils import get_data

# Make model name
setup = [
    ('model={:s}',  'classifier'),
    ('src={:s}',  args.src),
    ('design={:s}', args.design),
    # ('lr={:.0e}',  args.lr),
    ('run={:04d}',   args.run)
]

model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

M = classifier()
M.sess.run(tf.global_variables_initializer())
src = get_data(args.src)
trg = None

train(M, src, trg,
      saver=None,
      has_disc=False,
      add_z=False,
      model_name=model_name)
