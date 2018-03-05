
# Modified to include the mnist download code.
# Modified to be more similar to simple mnist.

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import argparse
import numpy as np
import sys
import os
import cntk as C
import mnist_utils as ut

from cntk.train import Trainer, minibatch_size_schedule 
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.device import cpu, try_set_default_device
from cntk.learners import sgd, adadelta, learning_parameter_schedule_per_sample
from cntk.ops import relu, element_times, constant, times
from cntk.layers import Dense, Sequential, For
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.train.training_session import *
from cntk.logging import ProgressPrinter, TensorBoardProgressWriter
from cntk.initializer import glorot_uniform, uniform

# Paths relative to current python file.
data_dir = os.path.dirname(os.path.abspath(__file__))

def check_path(path):
    if not os.path.exists(path):
        readme_file = os.path.normpath(os.path.join(
            os.path.dirname(path), "..", "README.md"))
        raise RuntimeError(
            "File '%s' does not exist. Please follow the instructions at %s to download and prepare it." % (path, readme_file))

def create_reader(path, is_training, input_dim, label_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='features', shape=input_dim, is_sparse=False),
        labels    = StreamDef(field='labels',   shape=label_dim, is_sparse=False)
    )), randomize=False, max_sweeps = INFINITELY_REPEAT if is_training else 1)

# Creates and trains a feedforward classification model for MNIST images
def convnet_mnist():
    
    # Set global device type.
    cpu = C.DeviceDescriptor.cpu_device()
    try_set_default_device(cpu, acquire_device_lock=False)
    
    # Define data.
    image_height = 28
    image_width  = 28
    num_channels = 1
    input_shape = (num_channels, image_height, image_width)
    input_dimensions = image_height * image_width * num_channels
    num_output_classes = 10

    # Input variables denoting the features and label data
    input_var = C.ops.input_variable(input_shape, np.float32)
    label_var = C.ops.input_variable(num_output_classes, np.float32)

    # Instantiate the feedforward classification model
    scaled_input = C.ops.element_times(C.ops.constant(0.00390625), input_var)

    # setup initializer
    init = uniform(scale= 0.1, seed=32)

    with C.layers.default_options(activation=C.ops.relu, pad=False):
        conv1 = C.layers.Convolution2D((5,5), 32, init=init, bias=False, pad=True)(scaled_input)
        pool1 = C.layers.MaxPooling((3,3), (2,2))(conv1)
        conv2 = C.layers.Convolution2D((3,3), 48, init=init, bias=False)(pool1)
        pool2 = C.layers.MaxPooling((3,3), (2,2))(conv2)
        conv3 = C.layers.Convolution2D((3,3), 64, init=init, bias=False)(pool2)
        dense4 = C.layers.Dense(96, init=init, bias=False)(conv3)
        drop4 = C.layers.Dropout(0.5, seed=32)(dense4)
        model = C.layers.Dense(num_output_classes, activation=None, init=init, bias=False)(drop4)
    
    # Define loss and error metric.
    ce = C.losses.cross_entropy_with_softmax(model, label_var)
    pe = C.metrics.classification_error(model, label_var)

    # Training config.
    minibatch_size = 64
    minibatch_iterations = 200

    # Instantiate progress writers.
    training_progress_output_freq = 100

    # Instantiate the trainer object to drive the model training.
    lr_schedule      = C.learning_parameter_schedule_per_sample(0.01)
    learner = C.learners.sgd(model.parameters, lr_schedule)
    trainer = C.Trainer(model, (ce, pe), learner)

    # Load train data
    path = os.path.normpath(os.path.join(data_dir, "Train-28x28_cntk_text.txt"))
    check_path(path)
    reader_train = create_reader(path, True, input_dimensions, num_output_classes)

    input_map = {
        input_var  : reader_train.streams.features,
        label_var  : reader_train.streams.labels
    }

    # Train model.
    for i in range(0, int(minibatch_iterations)):
        mb = reader_train.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)
        if (((i + 1) % training_progress_output_freq) == 0 and trainer.previous_minibatch_sample_count != 0):
            trainLossValue = trainer.previous_minibatch_loss_average
            evaluationValue = trainer.previous_minibatch_evaluation_average
            print("Minibatch:", i + 1, "CrossEntropyLoss = ", trainLossValue, "EvaluationCriterion = ", evaluationValue)

    
    # Load test data.
    path = os.path.normpath(os.path.join(data_dir, "Test-28x28_cntk_text.txt"))
    check_path(path)
    reader_test = create_reader(path, False, input_dimensions, num_output_classes)

    input_map = {
        input_var : reader_test.streams.features,
        label_var  : reader_test.streams.labels
    }

    # Test data for trained model.
    test_minibatch_size = 1
    num_test_samples = 10000
    test_result = 0.0
    for i in range(0, int(num_test_samples)):
        mb = reader_test.next_minibatch(test_minibatch_size, input_map=input_map)
        eval_error = trainer.test_minibatch(mb)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    return test_result / num_test_samples


if __name__=='__main__':

    trainPath = r'./Train-28x28_cntk_text.txt' 
    if not os.path.isfile(trainPath):
        os.chdir(os.path.abspath(os.path.dirname(__file__)))
        train = ut.load('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 60000)
        print ('Writing train text file...')
        ut.savetxt(trainPath, train)
        print ('Done.')

    testPath = r'./Test-28x28_cntk_text.txt'
    if not os.path.isfile(testPath):
        test = ut.load('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 10000)
        print ('Writing test text file...')
        ut.savetxt(testPath, test)
        print ('Done.')

    error = convnet_mnist()
    print("Test Error: %f" % error)