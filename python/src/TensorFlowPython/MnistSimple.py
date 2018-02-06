# https://www.tensorflow.org/get_started/mnist/beginners
# Modified slightly to have intermediate variables
# And currently new cost since SoftmaxCrossEntropyWithLogits
# is not supported via C++ API yet

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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  m = tf.matmul(x, W)
  y = m + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  # softmax_cross_entropy_with_logits is not supported in C++ API yet
  #perOutputLoss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
  perOutputLoss = tf.squared_difference(x=y, y=y_) # Note order is reversed
  cross_entropy = tf.reduce_mean(perOutputLoss)
  learningRate = 0.01
  optimizer = tf.train.GradientDescentOptimizer(learningRate)
  train_step = optimizer.minimize(cross_entropy)

  graph_location = '../../outputs/tf/MnistSimple/'
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location, tf.get_default_graph())
  
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  train = mnist.train
  batchSize = 64
  for _ in range(200):
    batch_xs, batch_ys = train.next_batch(batchSize, shuffle=False)
    batchRun = sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  r = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})
  print('{:.16f}'.format(r))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
