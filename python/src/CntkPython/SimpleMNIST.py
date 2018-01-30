# Modified to include the mnist download code.
# Modified to use the raw API and be more similar to the corresponding tensorflow example.

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
from cntk.learners import adadelta, learning_parameter_schedule_per_sample
from cntk.ops import relu, element_times, constant, times
from cntk.layers import Dense, Sequential, For
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.train.training_session import *
from cntk.logging import ProgressPrinter, TensorBoardProgressWriter

abs_path = os.path.dirname(os.path.abspath(__file__))

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
    )), randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)


# Creates and trains a feedforward classification model for MNIST images

def simple_mnist(tensorboard_logdir=None):
    input_dim = 784
    num_output_classes = 10
    
    # Input variables denoting the features and label data
    x = C.input_variable(input_dim, np.float32)
    y = C.input_variable(num_output_classes, np.float32) # Ideally input should be scaled, like the original example.

    # Model Parameters
    W = C.Parameter([input_dim, num_output_classes], init=0.0, name='W')
    b = C.Parameter(num_output_classes, init=0.0, name='b')
    m = times(x, W)         
    # Linear Model
    z = m + b

    # Define loss and optimizer
    ce = cross_entropy_with_softmax(z, y)
    pe = classification_error(z, y)

    data_dir = abs_path

    path = os.path.normpath(os.path.join(data_dir, "Train-28x28_cntk_text.txt"))
    check_path(path)

    reader_train = create_reader(path, True, input_dim, num_output_classes)

    input_map = {
        x  : reader_train.streams.features,
        y  : reader_train.streams.labels
    }

    # Training config
    minibatch_size = 64
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 10

    # Instantiate progress writers.
    #training_progress_output_freq = 100
    progress_writers = [ProgressPrinter(
        #freq=training_progress_output_freq,
        tag='Training',
        num_epochs=num_sweeps_to_train_with)]

    if tensorboard_logdir is not None:
        progress_writers.append(TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=z))

    # Instantiate the trainer object to drive the model training
    lr = learning_parameter_schedule_per_sample(1)
    trainer = Trainer(z, (ce, pe), adadelta(z.parameters, lr), progress_writers)  

    training_session(
        trainer=trainer,
        mb_source = reader_train,
        mb_size = minibatch_size,
        model_inputs_to_streams = input_map,
        max_samples = num_samples_per_sweep * num_sweeps_to_train_with,
        progress_frequency=num_samples_per_sweep
    ).train()
    
    # Load test data
    path = os.path.normpath(os.path.join(data_dir, "Test-28x28_cntk_text.txt"))
    check_path(path)

    reader_test = create_reader(path, False, input_dim, num_output_classes)

    input_map = {
        x  : reader_test.streams.features,
        y  : reader_test.streams.labels
    }

    # Test data for trained model
    test_minibatch_size = 1024
    num_samples = 10000
    num_minibatches_to_test = num_samples / test_minibatch_size
    test_result = 0.0
    for i in range(0, int(num_minibatches_to_test)):
        mb = reader_test.next_minibatch(test_minibatch_size, input_map=input_map)
        eval_error = trainer.test_minibatch(mb)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    return test_result / num_minibatches_to_test


if __name__=='__main__':
    # Specify the target device to be used for computing, if you do not want to
    # use the best available one, e.g.
    # try_set_default_device(cpu())

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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir',
                        help='Directory where TensorBoard logs should be created', required=False, default=None)
    args = vars(parser.parse_args())

    error = simple_mnist(args['tensorboard_logdir'])
    print("Error: %f" % error)