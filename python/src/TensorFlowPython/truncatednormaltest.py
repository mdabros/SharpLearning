# https://www.tensorflow.org/get_started/mnist/pros
# Modified slightly to have intermediate variables

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from tensorflow.python.ops import gen_random_ops

FLAGS = None

# There are two seeds, a graph level seed and a op level seed
GraphGlobalSeed = 17
OpGlobalSeed = 42

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1, seed=OpGlobalSeed)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  shape = [2, 3]
  tf.set_random_seed(GraphGlobalSeed)
  truncatedNormalDirect = gen_random_ops._truncated_normal(shape, tf.float32, seed=GraphGlobalSeed, seed2=OpGlobalSeed)
  
  w = weight_variable(shape)

  with tf.Session() as sess:
    truncatedNormalDirectPrint = tf.Print(truncatedNormalDirect, [truncatedNormalDirect])
    print(truncatedNormalDirectPrint.eval())

    globalVars = tf.global_variables()
    initializor = tf.variables_initializer(globalVars)
    initializeOutpu = sess.run(initializor)

    #globalVariables = tf.global_variables()
    for g in globalVars:
        # .initial_value()
        gp = tf.Print(g, [g])
        print(gp.eval())

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
