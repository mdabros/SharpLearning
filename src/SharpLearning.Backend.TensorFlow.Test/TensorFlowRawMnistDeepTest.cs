using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Backend.Testing;
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
        const int GlobalSeed = 42;

        [TestMethod]
        public void MnistDeep()
        {
            Assert.AreEqual(784, FeatureCount);

            using (var g = new TFGraph())
            {
                TFOutput x = g.Placeholder(TFDataType.Float, new TFShape(-1, FeatureCount), "x");
                TFOutput y_ = g.Placeholder(TFDataType.Float, new TFShape(-1, ClassCount), "y_");

                //# Build the graph for the deep net
                var (variablesList, y_conv, keep_prob) = DeepNeuralNet(g, x);

                //with tf.name_scope('loss'):
                TFOutput perOutputLoss = g.SquaredDifference(y_conv, y_, "diff");
                TFOutput loss = g.ReduceMean(perOutputLoss, operName: "reducemean");
                //cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_,
                //                                            logits = y_conv)
                //cross_entropy = tf.reduce_mean(cross_entropy)

                // with tf.name_scope('optimizer'):
                TFOutput learningRate = g.Const(new TFTensor(0.01f));
                // TODO: How to get variables dynamically from graph?
                TFOutput[] variablesOutputs = GetVariableOutputs(variablesList);
                TFOutput[] gradients = g.AddGradients(new TFOutput[] { loss }, variablesOutputs);

                TFOutput[] updates = new TFOutput[gradients.Length];
                for (int i = 0; i < gradients.Length; i++)
                {
                    updates[i] = g.ApplyGradientDescent(variablesOutputs[i], learningRate, gradients[i]);
                }
                //learningRate = 0.01
                //optimizer = tf.train.GradientDescentOptimizer(learningRate)
                //# We do not have AdamOptimizer in C# yet
                //#optimizer = tf.train.AdamOptimizer(1e-4)
                //train_step = optimizer.minimize(cross_entropy)

                TFOperation[] variablesAssignOps = GetVariableAssignOps(variablesList);

                TFOutput correctPrediction;
                TFOutput accuracy;
                //with tf.name_scope('accuracy'):
                using (g.WithScope("accuracy"))
                {
                    TFOutput one = g.Const(new TFTensor(1));
                    g.ArgMax(y_conv, one);
                    TFOutput argMaxActual = g.ArgMax(y_conv, one);
                    TFOutput argMaxExpected = g.ArgMax(y_, one);
                    correctPrediction = g.Equal(argMaxActual, argMaxExpected);
                    TFOutput castCorrectPrediction = g.Cast(correctPrediction, TFDataType.Float);
                    TFOutput correctSum = g.ReduceSum(castCorrectPrediction);
                    accuracy = g.ReduceMean(castCorrectPrediction);
                    //correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
                    //correct_prediction = tf.cast(correct_prediction, tf.float32)
                    //accuracy = tf.reduce_mean(correct_prediction)
                }
                //graph_location = tempfile.mkdtemp()
                //print('Saving graph to: %s' % graph_location)
                //train_writer = tf.summary.FileWriter(graph_location)
                //train_writer.add_graph(tf.get_default_graph())

                var data = DataSets.Mnist.Load(DownloadPath);

                const int trainBatchSize = 64;
                const int iterations = 100;

                var s = new Stopwatch();
                s.Start();

                using (var status = new TFStatus())
                using (var session = new TFSession(g))
                {
                    TFOperation[] targets = null; // TODO: It is possible to create a single operation target, instead of using outputs... how?
                    TFBuffer runMetaData = null;
                    TFBuffer runOptions = null;
                    TFStatus trainStatus = new TFStatus();

                    // Initialize variables
                    session.GetRunner().AddTarget(variablesAssignOps).Run();

                    // Dump initial assigns to see random initializes are identical
                    TFTensor[] variablesInitialized = session.Run(new TFOutput[] { }, new TFTensor[] { }, variablesOutputs,
                        targets, runMetaData, runOptions, trainStatus);

                    // Can be used for debugging initialization of all variables
                    //using (var w = new StreamWriter("MnistDeepVariablesInitial.txt"))
                    //{
                    //    foreach (var v in variablesInitialized)
                    //    {
                    //        var vText = v.ToString();
                    //        var array = (Array)v.GetValue();
                    //        w.WriteLine(vText);
                    //        w.WriteLine(array.ToDebugText());
                    //        Log(vText + " Initialize Logged");
                    //    }
                    //}

                    var initializeTime_ms = s.Elapsed.TotalMilliseconds;
                    s.Restart();

                    TFTensor train_dropout_keep_prob = new TFTensor(0.5f);
                    TFTensor test_dropout_keep_prob = new TFTensor(1f);

                    // Train (note that by using session.Run directly
                    //        we get a much more efficient loop, and it is easy to see what actually happens)
                    TFOutput[] inputs = new[] { x, y_, keep_prob };
                    TFOutput[] trainOutputs = updates; // Gradient updates are currently the outputs ensuring these are applied

                    TFOutput[] accuracyOutputs = new[] { accuracy };
                    TFOutput[] correctPredictionOutputs = new[] { correctPrediction };

                    TFTensor[] inputValues = new TFTensor[inputs.Length];


                    // Pre-allocate train tensors
                    TFTensor trainInputBatchTensor = new TFTensor(TFDataType.Float,
                        new long[] { trainBatchSize, FeatureCount }, sizeof(float) * trainBatchSize * FeatureCount);
                    TFTensor trainLabelBatchTensor = new TFTensor(TFDataType.Float,
                        new long[] { trainBatchSize, ClassCount }, sizeof(float) * trainBatchSize * ClassCount);

                    var rawTrainBatchEnumerator = data.CreateTrainBatchEnumerator(trainBatchSize);
                    var trainBatchEnumerator = Convert(rawTrainBatchEnumerator, ClassCount);
                    for (int i = 0; i < iterations && trainBatchEnumerator.MoveNext(); i++)
                    {
                        (float[] inputBatch, float[] labelBatch) = trainBatchEnumerator.CurrentBatch();

                        Marshal.Copy(inputBatch, 0, trainInputBatchTensor.Data, inputBatch.Length);
                        Marshal.Copy(labelBatch, 0, trainLabelBatchTensor.Data, labelBatch.Length);
                        inputValues[0] = trainInputBatchTensor;
                        inputValues[1] = trainLabelBatchTensor;
                        //inputValues[0] = TFTensor.FromBuffer(new TFShape(trainBatchSize, FeatureCount), inputBatch, 0, inputBatch.Length);
                        //inputValues[1] = TFTensor.FromBuffer(new TFShape(trainBatchSize, ClassCount), labelBatch, 0, labelBatch.Length);
                        
                        if (i % 100 == 0)
                        {
                            inputValues[2] = train_dropout_keep_prob;
                            TFTensor[] accuracyOutputValues = session.Run(inputs, inputValues, accuracyOutputs,
                                targets, runMetaData, runOptions, trainStatus);

                            trainStatus.Raise();

                            var train_accuracy = accuracyOutputValues[0];
                            float train_acc = (float)train_accuracy.GetValue();

                            Log($"Step {i} Accuracy {train_acc}");
                        }

                        inputValues[2] = train_dropout_keep_prob;
                        TFTensor[] outputValues = session.Run(inputs, inputValues, trainOutputs,
                            targets, runMetaData, runOptions, trainStatus);

                        trainStatus.Raise();
                    }
                    var trainTime_ms = s.Elapsed.TotalMilliseconds;
                    s.Restart();

                    // Test trained model
                    // Batch evaluation is faster (2x) and uses A LOT less memory, not batched == 6GB, batched == 240MB!
                    var testBatchSize = 100;
                    Debug.Assert(data.TestTargets.Data.Length % testBatchSize == 0); // Or we need to do something that handles remaining samples

                    // Pre-allocate train tensors
                    TFTensor testInputBatchTensor = new TFTensor(TFDataType.Float,
                        new long[] { testBatchSize, FeatureCount }, sizeof(float) * testBatchSize * FeatureCount);
                    TFTensor testLabelBatchTensor = new TFTensor(TFDataType.Float,
                        new long[] { testBatchSize, ClassCount }, sizeof(float) * testBatchSize * ClassCount);

                    var rawTestBatchEnumerator = data.CreateTestBatchEnumerator(testBatchSize);
                    var testBatchEnumerator = Convert(rawTestBatchEnumerator, ClassCount);
                    var totalCount = 0;
                    var correctCount = 0;
                    while (testBatchEnumerator.MoveNext())
                    {
                        var (testInputBatch, testLabelBatch) = testBatchEnumerator.CurrentBatch();

                        Marshal.Copy(testInputBatch, 0, testInputBatchTensor.Data, testInputBatch.Length);
                        Marshal.Copy(testLabelBatch, 0, testLabelBatchTensor.Data, testLabelBatch.Length);
                        inputValues[0] = testInputBatchTensor;
                        inputValues[1] = testLabelBatchTensor;
                        //inputValues[0] = TFTensor.FromBuffer(new TFShape(testBatchSize, FeatureCount), testInputBatch, 0, testInputBatch.Length);
                        //inputValues[1] = TFTensor.FromBuffer(new TFShape(testBatchSize, ClassCount), testLabelBatch, 0, testLabelBatch.Length);

                        inputValues[2] = test_dropout_keep_prob;

                        TFTensor[] outputValues = session.Run(inputs, inputValues, correctPredictionOutputs,
                            targets, runMetaData, runOptions, trainStatus);
                        
                        trainStatus.Raise();
                        TFTensor evaluatedCorrectPrediction = outputValues[0];

                        var ps = (bool[])evaluatedCorrectPrediction.GetValue();

                        totalCount += ps.Length;
                        foreach (var p in ps)
                        {
                            if (p) { ++correctCount; }
                        }
                    }

                    var incorrectCount = totalCount - correctCount;
                    var acc = correctCount / (float)totalCount;

                    var testTime_ms = s.Elapsed.TotalMilliseconds;
                    s.Restart();

                    // NOTE: That for even this simple CNN and for very small images the test time is about 1ms per image. I.e. 10000 ms. Depending on machine/CPU.
                    Log($"Accuracy {acc} Initialize {initializeTime_ms,6:F1} Train {trainTime_ms,6:F1} Test {testTime_ms,6:F1} [ms]");
                    Assert.AreEqual(0.0979999974370003, acc, 0.00000001); // This is what C# currently computes
                    Assert.AreEqual(0.2461999952793121, acc); // This is what equivalent python computes
                }
            }
        }

        private static IFlatBatchFeaturesTargetEnumerator<float, float> Convert(
            IFlatBatchFeaturesTargetEnumerator<byte, byte> enumerator, int classCount)
        {
            var convertEnumerator = enumerator
                // TODO: We really want to do this before batch enumerator so same as for tests..
                //.From().To(fb => (float)fb, tb => (int)tb);
                //.Feature().To(fb => (float)fb)
                .Feature().To(fb => fb * (1.0f / byte.MaxValue))
                .Target().ToOneHot(t => t, 1.0f, classCount);
            return convertEnumerator;
        }

        private TFOutput[] GetVariableOutputs(List<(TFOutput assign, TFOutput variable)> variablesList)
        {
            var outputs = new TFOutput[variablesList.Count];
            for (int i = 0; i < variablesList.Count; i++)
            {
                outputs[i] = variablesList[i].variable;
            }
            return outputs;
        }

        private TFOperation[] GetVariableAssignOps(List<(TFOutput assign, TFOutput variable)> variablesList)
        {
            var ops = new TFOperation[variablesList.Count];
            for (int i = 0; i < variablesList.Count; i++)
            {
                ops[i] = variablesList[i].assign.Operation;
            }
            return ops;
        }

        private (List<(TFOutput assign, TFOutput variable)> variables, TFOutput y_conv, TFOutput keep_prob) DeepNeuralNet(TFGraph g, TFOutput x)
        {
            //  """deepnn builds the graph for a deep net for classifying digits.
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

            var variables = new List<(TFOutput assign, TFOutput variable)>();

            const float bias = 0.1f;

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
                var W_conv1 = WeightVariable(g, new TFShape(5, 5, 1, FeatureMapsCount1));
                var b_conv1 = BiasVariable(g, bias, FeatureMapsCount1);
                variables.Add(W_conv1);
                variables.Add(b_conv1);
                TFOutput c_conv1 = Conv2D(g, x_image, W_conv1.variable, strides, Padding);
                TFOutput a_conv1 = g.Add(c_conv1, b_conv1.variable);
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
                var W_conv2 = WeightVariable(g, new TFShape(5, 5, FeatureMapsCount1, FeatureMapsCount2));
                var b_conv2 = BiasVariable(g, bias, FeatureMapsCount2);
                variables.Add(W_conv2);
                variables.Add(b_conv2);
                TFOutput c_conv2 = Conv2D(g, h_pool1, W_conv2.variable, strides, Padding);
                TFOutput a_conv2 = g.Add(c_conv2, b_conv2.variable);
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
            using (g.WithScope("fc1"))
            {
                var poolSizeDims = ImageSize / (2 * 2);
                var poolSize2 = poolSizeDims * poolSizeDims * FeatureMapsCount2;
                var W_fc1 = WeightVariable(g, new TFShape(poolSize2, FullyConnectedFeatures));
                var b_fc1 = BiasVariable(g, bias, FullyConnectedFeatures);
                variables.Add(W_fc1);
                variables.Add(b_fc1);
                TFOutput flatShape = g.Const(new TFShape(-1, poolSize2).AsTensor());
                TFOutput h_pool2_flat = g.Reshape(h_pool2, flatShape);
                TFOutput h_matmul = g.MatMul(h_pool2_flat, W_fc1.variable);
                TFOutput h_matmulbias = g.Add(h_matmul, b_fc1.variable);
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
                keep_prob = g.Placeholder(TFDataType.Float, new TFShape(1));
                var shape = new TFShape(FullyConnectedFeatures);
                h_fc1_drop = g.Dropout(h_fc1, keep_prob, shape, seed: GlobalSeed, operName: "dropOut");
            }
            //            with tf.name_scope('dropout'):
            //    keep_prob = tf.placeholder(tf.float32)
            //    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            //  # Map the 1024 features to 10 classes, one for each digit
            TFOutput y_conv;
            using (g.WithScope("fc2"))
            {
                var W_fc2 = WeightVariable(g, new TFShape(FullyConnectedFeatures, ClassCount));
                var b_fc2 = BiasVariable(g, 0.1f, ClassCount);
                variables.Add(W_fc2);
                variables.Add(b_fc2);
                TFOutput y_matmul = g.MatMul(h_fc1_drop, W_fc2.variable);
                y_conv = g.Add(y_matmul, b_fc2.variable);
            }
            //            with tf.name_scope('fc2'):
            //    W_fc2 = weight_variable([1024, 10])
            //    b_fc2 = bias_variable([10])
            //    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            return (variables, y_conv, keep_prob);
        }

        //def conv2d(x, W):
        //  """conv2d returns a 2d convolution layer with full stride."""
        //  return tf.nn.conv2d(x, W, strides =[1, 1, 1, 1], padding = 'SAME')
        public static TFOutput Conv2D(TFGraph g, TFOutput input, TFOutput filter, long[] strides, string padding="SAME")
        {
            return g.Conv2D(input, filter, strides, padding, data_format: "NHWC");
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

        // From TFS
        //public TFOutput DropoutImpl(TFOutput x, TFOutput keep_prob, TFShape noise_shape = null, int? seed = null, string operName = null)
        //{
        //    var scopeName = MakeName("dropout", operName);

        //    using (var newScope = WithScope(scopeName))
        //    {
        //        if (noise_shape == null)
        //            noise_shape = new TFShape(GetShape(x));

        //        TFOutput shapeTensor = ShapeTensorOutput(noise_shape);

        //        // uniform [keep_prob, 1.0 + keep_prob)
        //        TFOutput random_tensor = keep_prob;
        //        random_tensor = Add(random_tensor, RandomUniform(shapeTensor, seed: seed, dtype: x.OutputType));

        //        // 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        //        TFOutput binary_tensor = Floor(random_tensor);
        //        TFOutput ret = Mul(Div(x, keep_prob), binary_tensor);
        //        SetTensorShape(ret, GetShape(x));
        //        return ret;
        //    }
        //}

        //def weight_variable(shape):
        //  """weight_variable generates a weight variable of a given shape."""
        //  initial = tf.truncated_normal(shape, stddev = 0.1)
        //  return tf.Variable(initial)
        public static (TFOutput assign, TFOutput variable) WeightVariable(TFGraph g, TFShape shape)
        {
            const float mean = 0.0f;
            const float stddev = 0.1f;
            TFOutput shape_output = g.Const(shape.AsTensor());
            TFOutput rnd = g.TruncatedNormal(shape_output, TFDataType.Float, seed: GlobalSeed);
            TFTensor mean_tensor = new TFTensor(mean);
            TFTensor stddev_tensor = new TFTensor(stddev);
            TFOutput mean_output = g.Const(mean_tensor);
            TFOutput stddev_output = g.Const(stddev_tensor);
            TFOutput mul = g.Mul(rnd, stddev_output);
            TFOutput value = g.Add(mul, mean_output);
            TFOutput initial = value;
            //TFOutput initial = g.ParameterizedTruncatedNormal(shape_output, TFDataType.Float, seed: GlobalSeed);
            //with ops.name_scope(name, "truncated_normal", [shape, mean, stddev]) as name:
            //shape_tensor = _ShapeTensor(shape)
            //mean_tensor = ops.convert_to_tensor(mean, dtype = dtype, name = "mean")
            //stddev_tensor = ops.convert_to_tensor(stddev, dtype = dtype, name = "stddev")
            //seed1, seed2 = random_seed.get_seed(seed)
            //rnd = gen_random_ops._truncated_normal(
            //    shape_tensor, dtype, seed = seed1, seed2 = seed2)
            //mul = rnd * stddev_tensor
            //value = math_ops.add(mul, mean_tensor, name = name)
            //return value

            // What about std dev???
            //Variable variable = g.Variable(initial);

            TFOutput w = g.VariableV2(shape, TFDataType.Float, operName: "W");
            TFOutput w_init = g.Assign(w, initial);
            
            return (w_init, w);
        }

        //def bias_variable(shape):
        //  """bias_variable generates a bias variable of a given shape."""
        //  initial = tf.constant(0.1, shape = shape)
        //  return tf.Variable(initial)
        public static (TFOutput assign, TFOutput variable) BiasVariable(TFGraph g, float bias, int length)
        {
            // Const initialization is so weird...
            var array = new float[length];
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = bias;
            }
            TFTensor tensor = new TFTensor(array);
            TFOutput initial = g.Const(tensor);

            TFShape shape = new TFShape(tensor.Shape);
            TFOutput b = g.VariableV2(shape, TFDataType.Float, operName: "b");
            TFOutput b_init = g.Assign(b, initial);

            return (b_init, b);
            //Variable variable = g.Variable(initial);
            //return variable;
        }
    }
}
