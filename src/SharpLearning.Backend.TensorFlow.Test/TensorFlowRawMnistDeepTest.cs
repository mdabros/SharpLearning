using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Runtime.CompilerServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorFlow;

namespace SharpLearning.Backend.TensorFlow.Test
{
    [TestClass]
    public class TensorFlowRawMnistDeepTest
    {
        const string DownloadPath = "MnistTest";
        static readonly Action<string> Log = t => { Trace.WriteLine(t); Console.WriteLine(t); };
        const int ImageSize = 28;
        const int FeatureCount = ImageSize * ImageSize;
        const int ClassCount = 10;

        [TestMethod]
        public void MnistDeep()
        {
            Assert.AreEqual(784, FeatureCount);

            using (var g = new TFGraph())
            {
                TFOutput x = g.Placeholder(TFDataType.Float, new TFShape(-1, FeatureCount), "x");
                TFOutput y_ = g.Placeholder(TFDataType.Float, new TFShape(-1, ClassCount), "y_");

                //mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

                //# Create the model
                //x = tf.placeholder(tf.float32, [None, 784])

                //# Define loss and optimizer
                //y_ = tf.placeholder(tf.float32, [None, 10])

                //# Build the graph for the deep net
                //y_conv, keep_prob = deepnn(x)
                var (y_conv, keep_prob) = DeepNeuralNet(g, x);

                //with tf.name_scope('loss'):
                //cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_,
                //                                            logits = y_conv)
                //cross_entropy = tf.reduce_mean(cross_entropy)

                //with tf.name_scope('adam_optimizer'):
                //train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

                //with tf.name_scope('accuracy'):
                //correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
                //correct_prediction = tf.cast(correct_prediction, tf.float32)
                //accuracy = tf.reduce_mean(correct_prediction)

                //graph_location = tempfile.mkdtemp()
                //print('Saving graph to: %s' % graph_location)
                //train_writer = tf.summary.FileWriter(graph_location)
                //train_writer.add_graph(tf.get_default_graph())

                //with tf.Session() as sess:
                //sess.run(tf.global_variables_initializer())
                //#for i in range(20000):
                //for i in range(200):
                //batch = mnist.train.next_batch(50)
                //if i % 100 == 0:
                //train_accuracy = accuracy.eval(feed_dict ={
                //    x: batch[0], y_: batch[1], keep_prob: 1.0})
                //print('step %d, training accuracy %g' % (i, train_accuracy))
                //train_step.run(feed_dict ={ x: batch[0], y_: batch[1], keep_prob: 0.5})

                //print('test accuracy %g' % accuracy.eval(feed_dict ={
                //    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

                TFOutput W_zero = g.Const(new float[FeatureCount, ClassCount]);
                TFOutput b_zero = g.Const(new float[ClassCount]);

                TFOutput W = g.VariableV2(new TFShape(FeatureCount, ClassCount), TFDataType.Float, "W");
                // Only way to simply set zeros??
                TFOutput W_init = g.Assign(W, W_zero);

                TFOutput b = g.VariableV2(new TFShape(ClassCount), TFDataType.Float, "b");
                TFOutput b_init = g.Assign(b, b_zero);


                TFOutput m = g.MatMul(x, W, operName: "xW");
                TFOutput y = g.Add(m, b, operName: "y");

                //// SoftmaxCrossEntropyWithLogits: gradient for this is not yet supported
                //// see: https://github.com/tensorflow/tensorflow/pull/14727
                ////var (perOutputLoss, backprop) = g.SoftmaxCrossEntropyWithLogits(y, expectedY, "softmax");
                //TFOutput perOutputLoss = g.SquaredDifference(y, expectedY, "softmax");
                //TFOutput loss = g.ReduceMean(perOutputLoss, operName: "reducemean");

                //TFOutput learningRate = g.Const(new TFTensor(0.01f));

                //// TODO: How to get variables dynamically?
                //TFOutput[] variables = new TFOutput[] { W, b };
                //TFOutput[] gradients = g.AddGradients(new TFOutput[] { loss }, variables);


                //TFOutput[] updates = new TFOutput[gradients.Length];
                //for (int i = 0; i < gradients.Length; i++)
                //{
                //    updates[i] = g.ApplyGradientDescent(variables[i], learningRate, gradients[i]);
                //}

                //var mnist = Mnist.Load(DownloadPath);
                //const int batchSize = 100;
                //const int iterations = 200;

                //using (var status = new TFStatus())
                //using (var session = new TFSession(g))
                //{
                //    // Initialize variables
                //    session.GetRunner().AddTarget(W_init.Operation).Run();
                //    session.GetRunner().AddTarget(b_init.Operation).Run();

                //    // Train (note that by using session.Run directly
                //    //        we get a much more efficient loop, and it is easy to see what actually happens)
                //    TFOutput[] inputs = new[] { x, expectedY };
                //    TFOutput[] outputs = updates; // Gradient updates are currently the outputs ensuring these are applied
                //    TFOperation[] targets = null; // TODO: It is possible to create a single operation target, instead of using outputs... how?
                    
                //    TFBuffer runMetaData = null;
                //    TFBuffer runOptions = null;
                //    TFStatus trainStatus = new TFStatus();

                //    var trainReader = mnist.GetTrainReader();
                //    for (int i = 0; i < iterations; i++)
                //    {
                //        (float[,] inputBatch, float[,] labelBatch) = trainReader.NextBatch(batchSize);

                //        TFTensor[] inputValues = new [] { new TFTensor(inputBatch), new TFTensor(labelBatch) };

                //        TFTensor[] outputValues = session.Run(inputs, inputValues, outputs, 
                //            targets, runMetaData, runOptions, trainStatus);
                //    }

                //    // Test trained model
                //    TFOutput one = g.Const(new TFTensor(1));
                //    TFOutput argMaxActual = g.ArgMax(y, one);
                //    TFOutput argMaxExpected = g.ArgMax(expectedY, one);
                //    TFOutput correctPrediction = g.Equal(argMaxActual, argMaxExpected);
                //    TFOutput castCorrectPrediction = g.Cast(correctPrediction, TFDataType.Float);
                //    TFOutput accuracy = g.ReduceMean(castCorrectPrediction);

                //    var testReader = mnist.GetTestReader();
                //    var (testImages, testLabels) = testReader.All();

                //    TFTensor evaluatedAccuracy = session.GetRunner()
                //        .AddInput(x, testImages)
                //        .AddInput(expectedY, testLabels)
                //        .Run(accuracy);

                //    float acc = (float)evaluatedAccuracy.GetValue();

                //    Log($"Accuracy {acc}");
                //    Assert.AreEqual(acc, 0.797);
                //}
            }
        }

        private (TFOutput y_conv, TFOutput keep_prob) DeepNeuralNet(TFGraph g, TFOutput x)
        {

            //            """deepnn builds the graph for a deep net for classifying digits.

            //  Args:
            //            x: an input tensor with the dimensions(N_examples, 784), where 784 is the
            //           number of pixels in a standard MNIST image.

            //  Returns:
            //            A tuple(y, keep_prob). y is a tensor of shape(N_examples, 10), with values
            //    equal to the logits of classifying the digit into one of 10 classes(the
            //    digits 0 - 9).keep_prob is a scalar placeholder for the probability of

            //     dropout.

            //  """
            // Reshape to use within a convolutional neural net.
            // Last dimension is for "features" - there is only one here, since images are
            // grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
            //  with tf.name_scope('reshape'):

            // Scope usage is weird in C#...
            TFOutput x_image;
            using (g.WithScope("reshape"))
            {
                TFShape shape = new TFShape(-1, ImageSize, ImageSize, 1);
                TFTensor shape_tensor = shape.AsTensor();
                TFOutput shape_output = g.Const(shape_tensor);
                x_image = g.Reshape(x, shape_output);
                //     x_image = tf.reshape(x, [-1, 28, 28, 1])
            }

            long[] strides = new long[] { 1, 1, 1, 1 };
            const string Padding = "SAME";
            const int FeatureMapsCount1 = 32;
            //  # First convolutional layer - maps one grayscale image to 32 feature maps.
            //  with tf.name_scope('conv1'):
            TFOutput h_conv1;
            using (g.WithScope("conv1"))
            {
                Variable W_conv1 = WeightVariable(g, new TFShape(5, 5, 1, FeatureMapsCount1));
                Variable b_conv1 = BiasVariable(g, 0.1f, FeatureMapsCount1);
                TFOutput c_conv1 = Conv2D(g, x_image, W_conv1, strides, Padding);
                TFOutput a_conv1 = g.Add(c_conv1, b_conv1);
                h_conv1 = g.Relu(a_conv1);
            }
            //     W_conv1 = weight_variable([5, 5, 1, 32])
            //   b_conv1 = bias_variable([32])
            //    c_conv1 = conv2d(x_image, W_conv1)
            //    a_conv1 = c_conv1 + b_conv1
            //    h_conv1 = tf.nn.relu(a_conv1)

            //  # Pooling layer - downsamples by 2X.
            //            with tf.name_scope('pool1'):
            //    h_pool1 = max_pool_2x2(h_conv1)
            TFOutput h_pool1 = MaxPool2x2(g, h_conv1);

            //  # Second convolutional layer -- maps 32 feature maps to 64.
            const int FeatureMapsCount2 = 64;
            TFOutput h_conv2;
            using (g.WithScope("conv2"))
            {
                Variable W_conv2 = WeightVariable(g, new TFShape(5, 5, FeatureMapsCount1, FeatureMapsCount2));
                Variable b_conv2 = BiasVariable(g, 0.1f, FeatureMapsCount2);
                TFOutput c_conv2 = Conv2D(g, h_pool1, W_conv2, strides, Padding);
                TFOutput a_conv2 = g.Add(c_conv2, b_conv2);
                h_conv2 = g.Relu(a_conv2);
            }
            //            with tf.name_scope('conv2'):
            //    W_conv2 = weight_variable([5, 5, 32, 64])
            //    b_conv2 = bias_variable([64])
            //    c_conv2 = conv2d(h_pool1, W_conv2)
            //    a_conv2 = c_conv2 + b_conv2
            //    h_conv2 = tf.nn.relu(a_conv2)

            //  # Second pooling layer.
            //            with tf.name_scope('pool2'):
            //    h_pool2 = max_pool_2x2(h_conv2)
            TFOutput h_pool2 = MaxPool2x2(g, h_conv2);

            //# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
            //# is down to 7x7x64 feature maps -- maps this to 1024 features.
            const int FullyConnectedFeatures = 1024;
            TFOutput h_fc1;
            using (g.WithScope("conv2"))
            {
                var poolSizeDims = ImageSize / (2 * 2);
                var poolSize2 = poolSizeDims * poolSizeDims * FeatureMapsCount2;
                Variable W_fc1 = WeightVariable(g, new TFShape(poolSize2, FullyConnectedFeatures));
                Variable b_fc1 = BiasVariable(g, 0.1f, FullyConnectedFeatures);
                TFOutput flatShape = g.Const(new TFShape(-1, poolSize2).AsTensor());
                TFOutput h_pool2_flat = g.Reshape(h_pool2, flatShape);
                TFOutput h_matmul = g.MatMul(h_pool2_flat, W_fc1);
                TFOutput h_matmulbias = g.Add(h_matmul, b_fc1);
                h_fc1 = g.Relu(h_matmulbias);
            }

            //            with tf.name_scope('fc1'):
            //    W_fc1 = weight_variable([7 * 7 * 64, 1024])
            //    b_fc1 = bias_variable([1024])

            //    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            //    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            //# Dropout - controls the complexity of the model, prevents co-adaptation of
            //# features.
            TFOutput h_fc1_drop;
            TFOutput keep_prob;
            using (g.WithScope("dropout"))
            {
                keep_prob = g.Placeholder(TFDataType.Float);
                h_fc1_drop = g.Dropout(h_fc1, keep_prob);
            }
            //            with tf.name_scope('dropout'):
            //    keep_prob = tf.placeholder(tf.float32)
            //    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            //  # Map the 1024 features to 10 classes, one for each digit
            TFOutput y_conv;
            using (g.WithScope("fc2"))
            {
                Variable W_fc2 = WeightVariable(g, new TFShape(FullyConnectedFeatures, ClassCount));
                Variable b_fc2 = BiasVariable(g, 0.1f, ClassCount);
                TFOutput y_matmul = g.MatMul(h_fc1_drop, W_fc2);
                y_conv = g.Add(y_matmul, b_fc2);
            }
            //            with tf.name_scope('fc2'):
            //    W_fc2 = weight_variable([1024, 10])
            //    b_fc2 = bias_variable([10])
            //    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            return (y_conv, keep_prob);
        }

        //def conv2d(x, W):
        //  """conv2d returns a 2d convolution layer with full stride."""
        //  return tf.nn.conv2d(x, W, strides =[1, 1, 1, 1], padding = 'SAME')
        public static TFOutput Conv2D(TFGraph g, TFOutput x, TFOutput W, long[] strides, string padding="SAME")
        {
            return g.Conv2D(x, W, strides, padding);
        }

        //def max_pool_2x2(x):
        //  """max_pool_2x2 downsamples a feature map by 2X."""
        //  return tf.nn.max_pool(x, ksize =[1, 2, 2, 1],
        //                        strides =[1, 2, 2, 1], padding = 'SAME')
        const string MaxPool2x2Padding = "SAME";
        static readonly long[] MaxPool2x2KernelSize = new long[] { 1, 2, 2, 1 };
        static readonly long[] MaxPool2x2Strides = new long[] { 1, 2, 2, 1 };
        private TFOutput MaxPool2x2(TFGraph g, TFOutput x)
        {
            return g.MaxPool(x, MaxPool2x2KernelSize, MaxPool2x2Strides, MaxPool2x2Padding);
        }


        //def weight_variable(shape):
        //  """weight_variable generates a weight variable of a given shape."""
        //  initial = tf.truncated_normal(shape, stddev = 0.1)
        //  return tf.Variable(initial)
        public static Variable WeightVariable(TFGraph g, TFShape shape)
        {
            TFOutput shape_output = g.Const(shape.AsTensor());
            TFOutput initial = g.TruncatedNormal(shape_output, TFDataType.Float);
            // What about std dev???
            Variable variable = g.Variable(initial);
            return variable;
        }

        //def bias_variable(shape):
        //  """bias_variable generates a bias variable of a given shape."""
        //  initial = tf.constant(0.1, shape = shape)
        //  return tf.Variable(initial)
        public static Variable BiasVariable(TFGraph g, float bias, int length)
        {
            // Const initialization is so weird...
            var array = new float[length];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = bias;
            }
            TFTensor tensor = new TFTensor(array);
            TFOutput output = g.Const(tensor);
            Variable variable = g.Variable(output);
            return variable;
        }
    }
}
