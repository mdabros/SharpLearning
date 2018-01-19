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
    public class TensorFlowRawMnistTest
    {
        const string DownloadPath = "MnistTest";
        static readonly Action<string> Log = t => { Trace.WriteLine(t); Console.WriteLine(t); };

        [TestMethod]
        public void MnistTest()
        {

            const int featureCount = 28 * 28;
            Assert.AreEqual(784, featureCount);
            const int classCount = 10;

            using (var g = new TFGraph())
            {
                //var scope = g.WithScope("tst");

                TFOutput x = g.Placeholder(TFDataType.Float, new TFShape(-1, featureCount), "x");
                TFOutput expectedY = g.Placeholder(TFDataType.Float, new TFShape(-1, classCount), "y_");

                TFOutput W_zero = g.Const(new float[featureCount, classCount]);
                TFOutput b_zero = g.Const(new float[classCount]);

                // Only way to simply set zeros??
                var W = g.VariableV2(new TFShape(featureCount, classCount), TFDataType.Float, "W");
                var W_init = g.Assign(W, W_zero);
                //TFOperation W_op_init;
                //TFOutput W_out_value;
                //var W = g.Variable(W_zero, out W_op_init, out W_out_value,  operName: "W");
                //g.ZerosLike( // Does not work with ApplyGradientDescent, how do we inialize??
                //g.Variable(new TFShape(featureCount, classCount), TFDataType.Float, "W")
                //)
                //;
                //g.Assign(W, w_zero);

                //TFOperation b_op_init;
                //TFOutput b_out_value;
                //Variable b = g.Variable(b_zero, out b_op_init, out b_out_value, operName: "b");
                //g.ZerosLike(
                //g.Variable(new TFShape(classCount), TFDataType.Float, "b")
                //)
                //;
                //g.Assign(b, b_zero);

                var b = g.VariableV2(new TFShape(classCount), TFDataType.Float, "b");
                var b_init = g.Assign(b, b_zero);


                var m = g.MatMul(x, W, operName: "xW");
                var y = g.Add(m, b, operName: "y");


                var (loss, backprop) = g.SoftmaxCrossEntropyWithLogits(y, expectedY, "softmax");
                var crossEntropy = g.ReduceMean(loss, operName: "reducemean");



                //g.variab(().Select(v => v.VariableOp).ToArray();

                // Is this right??
                //var b_gradient = g.BiasAddGrad(backprop);
                // Is this right??
                var variables = new TFOutput[] { W, b };
                var gradients = g.AddGradients(new TFOutput[] { y }, variables);

                var learningRate = g.Const(new TFTensor(0.5f));

                var applieds = new TFOutput[gradients.Length];
                for (int i = 0; i < gradients.Length; i++)
                {
                    applieds[i] = g.ApplyGradientDescent(variables[i], learningRate, gradients[i]);
                }
                //var applieds = new TFOperation[gradients.Length];
                //for (int i = 0; i < gradients.Length; i++)
                //{
                //    applieds[i] = g.ResourceApplyGradientDescent(variables[i], learningRate, gradients[i]);
                //}

                // Need to do this in loop, can't do it before we have actual variables...
                // TODO: Is this how to do it???
                // https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/apply-gradient-descent
                //var adjustedW = g.ApplyGradientDescent(W, learningRate, backprop);
                // See https://github.com/tensorflow/tensorflow/pull/11377#issuecomment-324546254
                // use ResourceApplyGradientDescent

                //var ops = g.GetEnumerator();
                //foreach(var op in ops)
                //{

                //}

                // How to adjust bias?
                //var adjustedB = g.ApplyGradientDescent(b, learningRate, backprop);
                // https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc/gradients

                // https://github.com/tensorflow/tensorflow/pull/11377

                // See https://blog.manash.me/implementation-of-gradient-descent-in-tensorflow-using-tf-gradients-c111d783c78b

                var mnist = Mnist.Load(DownloadPath);
                var trainReader = mnist.GetTrainReader();
                const int batchSize = 64;
                const int trainingEpochs = 20;

                //g.AddInitVariable(W);
                using (var status = new TFStatus())
                using (var s = new TFSession(g))
                {
                    // initialize variables
                    s.GetRunner().AddTarget(W_init.Operation).Run();
                    s.GetRunner().AddTarget(b_init.Operation).Run();

                    // https://stackoverflow.com/questions/46760202/how-to-initialize-variables-in-tensorflow-c-api
                    // This C API
                    //TF_Operation* init_op = TF_GraphGetOperationByName(graph, "init");
                    //TF_SessionRun(sess, NULL,
                    //              NULL, NULL, 0,  // inputs
                    //              NULL, NULL, 0,  // outputs
                    //              &init_op, 1,    // targets
                    //              NULL,
                    //              status);

                    // https://github.com/migueldeicaza/TensorFlowSharp/issues/170

                    //var initValue = g.Const(1.5);
                    //var increment = g.Const(0.5);
                    //TFOperation init;
                    //TFOutput value;
                    //var handle = g.Variable(initValue, out init, out value);

                    // Must first initialize all the variables.
                    //s.GetRunner().AddTarget(x).Run(status);
                    //Assert(status);

                    // Can this be used to inialize all variables...
                    //TFOperation[] globalVariables = g.GetGlobalVariablesInitializer();
                    //for (int i = 0; i < globalVariables.Length; i++)
                    //{
                    //    s.GetRunner().AddTarget(globalVariables[i]).Run();
                    //}
                    for (int epoch = 0; epoch < trainingEpochs; epoch++)
                    {
                        //s.Run(ops, new TFTensor[] { x, expectedY }, null);

                        //foreach (var observation in observations)
                        //{

                        //    s.Run(ops, new TFTensor[] { observation.X, observation.Y }, null);
                        //}
                        var (inputBatch, labelBatch) = trainReader.NextBatch(batchSize);

                        var c = s.GetRunner()
                            .AddInput(x, inputBatch)
                            .AddInput(expectedY, labelBatch)
                            .Run(crossEntropy);
                        AssertStatus(status);


                        // Display logs per epoch step
                        //if ((epoch + 1) % display_step == 0)
                        {
                            //var c = runner
                            //    .AddInput(X, train_x)
                            //    .AddInput(Y, train_y)
                            //    .Run(cost);
                            // , W={runner.Run(W)}, b={runner.Run(b)}
                            Log($"Epoch: {epoch + 1}, cost={c}");
                        }
                    }
                }
            }

        }

        [TestMethod]
        public void Run_TensorFlow_Logistic_Regression()
        {
            Trace.WriteLine("Logistic regression");
            // Parameters
            var learning_rate = 0.001f;
            var training_epochs = 1000;
            var display_step = 50;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };

            var rnd = new Random(23);
            // random 0 or 1 for logistic regression targets.
            var train_y = train_x.Select(v => (float)rnd.Next(2)).ToArray();

            var n_samples = train_x.Length;

            var dataType = TFDataType.Float;

            using (var g = new TFGraph())
            {
                var s = new TFSession(g);

                var X = g.Placeholder(dataType);
                var Y = g.Placeholder(dataType);

                var W = g.VariableV2(TFShape.Scalar, dataType, operName: "W");
                var initW = g.Assign(W, g.Const((float)rnd.NextDouble()));

                var b = g.VariableV2(TFShape.Scalar, dataType, operName: "b");
                var initb = g.Assign(b, g.Const((float)rnd.NextDouble()));
                var param = new[] { W, b };

                var pred = g.Sigmoid(g.Add(g.Mul(X, W), b));

                // [WIP] Tensorflow c++ API still missing full gradient support.
                // BinaryCrossEntropy loss.
                var loss = g.ReduceMean(g.SigmoidCrossEntropyWithLogits(pred, Y));
                var gradients = g.AddGradients(new[] { loss }, param);

                // figure out how to do updates on lists of params and gradients.
                var updateW = g.Assign(W, g.Add(W, g.Mul(g.Const(-learning_rate), gradients[0])));
                var updateB = g.Assign(b, g.Add(b, g.Mul(g.Const(-learning_rate), gradients[1])));

                // initialize variables
                s.GetRunner().AddTarget(initW.Operation).Run();
                s.GetRunner().AddTarget(initb.Operation).Run();

                var observations = train_x.Zip(train_y, (x, y) => new { X = x, Y = y }).ToArray();
                for (int epoch = 0; epoch < training_epochs; epoch++)
                {
                    foreach (var observation in observations)
                    {

                        // run optimization loop. Minimization does not seem to work properly [WIP].
                        s.GetRunner()
                        .Fetch(updateB)
                        .Fetch(updateW)
                        .Fetch(gradients)
                        .Fetch(loss)
                        .Fetch(pred)
                        .AddInput(X, observation.X)
                        .AddInput(Y, observation.Y)
                        .Run();
                    }

                    // Display logs per epoch step
                    if ((epoch + 1) % display_step == 0)
                    {
                        var c = s.GetRunner()
                            .AddInput(X, train_x)
                            .AddInput(Y, train_y)
                            .Run(loss);

                        Trace.WriteLine("Epoch: " + (epoch + 1) + ", cost=" + c + ", W=" + s.GetRunner().Run(W).GetValue() + ", b=" + s.GetRunner().Run(b));
                    }
                }
            }
        }

        [TestMethod]
        public void TestVariable()
        {
            Console.WriteLine("Variables");
            var status = new TFStatus();
            using (var g = new TFGraph())
            {
                var initValue = g.Const(1.5);
                var increment = g.Const(0.5);
                TFOperation init;
                TFOutput value;
                var handle = g.Variable(initValue, out init, out value);

                // Add 0.5 and assign to the variable.
                // Perhaps using op.AssignAddVariable would be better,
                // but demonstrating with Add and Assign for now.
                var update = g.AssignVariableOp(handle, g.Add(value, increment));

                var s = new TFSession(g);
                // Must first initialize all the variables.
                s.GetRunner().AddTarget(init).Run(status);
                AssertStatus(status);
                // Now print the value, run the update op and repeat
                // Ignore errors.
                for (int i = 0; i < 5; i++)
                {
                    // Read and update
                    var result = s.GetRunner().Fetch(value).AddTarget(update).Run();

                    Console.WriteLine("Result of variable read {0} -> {1}", i, result[0].GetValue());
                }
            }
        }

        public static void AssertStatus(TFStatus status, [CallerMemberName] string caller = null, string message = "")
        {
            if (status.StatusCode != TFCode.Ok)
            {
                throw new Exception($"{caller}: {status.StatusMessage} {message}");
            }
        }

        //nist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

        //  # Create the model
        //            x = tf.placeholder(tf.float32, [None, 784])
        //  W = tf.Variable(tf.zeros([784, 10]))
        //  b = tf.Variable(tf.zeros([10]))
        //  y = tf.matmul(x, W) + b

        //  # Define loss and optimizer
        //            y_ = tf.placeholder(tf.float32, [None, 10])

        //  # The raw formulation of cross-entropy,
        //#
        //# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
        //# reduction_indices=[1]))
        //#
        //# can be numerically unstable.
        //#
        //# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
        //# outputs of 'y', and then average across the batch.
        //            cross_entropy = tf.reduce_mean(
        //      tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))
        //  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        //  sess = tf.InteractiveSession()
        //  tf.global_variables_initializer().run()
        //  # Train
        //  for _ in range(1000):
        //    batch_xs, batch_ys = mnist.train.next_batch(100)
        //    sess.run(train_step, feed_dict ={ x: batch_xs, y_: batch_ys})

        //  # Test trained model
        //  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        //  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        //  print(sess.run(accuracy, feed_dict ={
        //                x: mnist.test.images,
        //                                      y_: mnist.test.labels}))

        // TEST_F(OptimizerTest, OneMatMul)
        //        {
        //        +  for (const bool expected : { false, true}) {
        //            +    const Scope&scope = expected ? scope_expected_ : scope_test_;
        //            +
        //            +    // the forward node should be the same for the test and expected scope
        //            +    // TODO(theflofly): merge Const and Assign using one
        //            +    // constructor as in python
        //            +auto x = Variable(scope.WithOpName("x"), { 2, 2}, DT_FLOAT);
        //            +auto const_x = Const(scope, { { 1.0f, 2.0f}, { 3.0f, 4.0f}
        //            });
        //            +auto assign_x = Assign(scope.WithOpName("Assign_x"), x, const_x);
        //            +
        //            +auto y = Variable(scope.WithOpName("y"), { 2, 2}, DT_FLOAT);
        //            +auto const_y = Const(scope, { { 1.0f, 0.0f}, { 0.0f, 1.0f}
        //            });
        //            +auto assign_y = Assign(scope.WithOpName("Assign_y"), y, const_y);
        //            +
        //            +    // the assign node is only used once, it should not be used in the graph
        //            +auto z = MatMul(scope.WithOpName("z"), x, y);
        //            +
        //            +TF_ASSERT_OK(scope.status());
        //            +CHECK_NOTNULL(z.node());
        //            +
        //            +    if (expected)
        //            {
        //                +      // we manually add the gradient node to the expected scope
        //                +Scope scope_gradient = scope.NewSubScope("Gradients");
        //                +Scope scope_optimizer = scope.NewSubScope("GradientDescent");
        //                +
        //                +      // gradients
        //                +auto dz = ops::OnesLike(scope_gradient, z);
        //                +auto dx = MatMul(scope_gradient, dz, y, MatMul::TransposeB(true));
        //                +auto dy = MatMul(scope_gradient, x, dz, MatMul::TransposeA(true));
        //                +
        //                +      // update
        //                +auto learning_rate1 =
        //            +Cast(scope_optimizer.NewSubScope("learning_rate"), 0.01f,
        //            +static_cast<DataType>(Output{ x}.type() - 100));
        //                +
        //                +ApplyGradientDescent(scope_optimizer.NewSubScope("update"), { x},
        //+learning_rate1, { dx});
        //                +
        //                +auto learning_rate2 =
        //            +Cast(scope_optimizer.NewSubScope("learning_rate"), 0.01f,
        //            +static_cast<DataType>(Output{ x}.type() - 100));
        //                +
        //                +ApplyGradientDescent(scope_optimizer.NewSubScope("update"), { y},
        //+learning_rate2, { dy});
        //                +    }
        //            else
        //            {
        //                +      // the gradient nodes and update nodes are added to the graph
        //                +auto train = GradientDescentOptimizer(0.01).Minimize(scope, { z});
        //                +
        //                +TF_ASSERT_OK(scope.status());
        //                +
        //                +ClientSession session(scope);
        //                +
        //                +      // TODO(theflofly): a global initializer would be nice
        //                +TF_CHECK_OK(session.Run({ assign_x, assign_y}, nullptr));

        //# Construct model
        //pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

        //# Minimize error using cross entropy
        //cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

        //grad_W, grad_b = tf.gradients(xs=[W, b], ys=cost)


        //new_W = W.assign(W - learning_rate * grad_W)
        //new_b = b.assign(b - learning_rate * grad_b)

        //# Initialize the variables (i.e. assign their default value)
        //init = tf.global_variables_initializer()

        //# Start training
        //with tf.Session() as sess:
        //    sess.run(init)

        //    # Training cycle
        //    for epoch in range(training_epochs):
        //        avg_cost = 0.
        //        total_batch = int(mnist.train.num_examples/batch_size)
        //        # Loop over all batches
        //        for i in range(total_batch):
        //            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        //            # Fit training using batch data
        //            _, _,  c = sess.run([new_W, new_b ,cost], feed_dict={x: batch_xs,
        //                                                       y: batch_ys})

        //            # Compute average loss
        //            avg_cost += c / total_batch
        //        # Display logs per epoch step
        //        if (epoch+1) % display_step == 0:
        //#             print(sess.run(W))
        //            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        //    print ("Optimization Finished!")

        //    # Test model
        //    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        //    # Calculate accuracy for 3000 examples
        //    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        //    print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))


        // NOT SURE WHAT THE HELL THE BELOW DOES, 
        // but mainly as code to see how TF works! Do not use for any real training.
        // BUG has been fixed by fixing loading
        // This sample has a bug, I suspect the data loaded is incorrect, because the returned
        // values in distance is wrong, and so is the prediction computed from it.
        //[TestMethod]
        public void NearestNeighbor()
        {
            // Get the Mnist data

            var mnist = Mnist.Load(DownloadPath);

            // 5000 for training
            const int trainCount = 5000;
            const int testCount = 200;
            (var trainingImages, var trainingLabels) = mnist.GetTrainReader().NextBatch(trainCount);
            (var testImages, var testLabels) = mnist.GetTestReader().NextBatch(testCount);

            Console.WriteLine("Nearest neighbor on Mnist images");
            using (var g = new TFGraph())
            {
                var s = new TFSession(g);


                TFOutput trainingInput = g.Placeholder(TFDataType.Float, new TFShape(-1, 784));

                TFOutput xte = g.Placeholder(TFDataType.Float, new TFShape(784));

                // Nearest Neighbor calculation using L1 Distance
                // Calculate L1 Distance
                TFOutput distance = g.ReduceSum(g.Abs(g.Add(trainingInput, g.Neg(xte))), axis: g.Const(1));

                // Prediction: Get min distance index (Nearest neighbor)
                TFOutput pred = g.ArgMin(distance, g.Const(0));

                var accuracy = 0f;
                // Loop over the test data
                for (int i = 0; i < testCount; i++)
                {
                    var runner = s.GetRunner();

                    // Get nearest neighbor

                    var result = runner.Fetch(pred).Fetch(distance).AddInput(trainingInput, trainingImages).AddInput(xte, Extract(testImages, i)).Run();
                    var r = result[0].GetValue();
                    var tr = result[1].GetValue();
                    var nn_index = (int)(long)result[0].GetValue();

                    // Get nearest neighbor class label and compare it to its true label
                    //Console.WriteLine($"Test {i}: Prediction: {ArgMax(trainingLabels, nn_index)} True class: {ArgMax(testLabels, i)} (nn_index={nn_index})");
                    if (ArgMax(trainingLabels, nn_index) == ArgMax(testLabels, i))
                        accuracy += 1f / testImages.Length;
                }
                Console.WriteLine("Accuracy: " + accuracy);
                Trace.WriteLine("Accuracy: " + accuracy);
            }
        }
        
        int ArgMax(float[,] array, int idx)
        {
            float max = -1;
            int maxIdx = -1;
            var l = array.GetLength(1);
            for (int i = 0; i < l; i++)
                if (array[idx, i] > max)
                {
                    maxIdx = i;
                    max = array[idx, i];
                }
            return maxIdx;
        }

        public float[] Extract(float[,] array, int index)
        {
            var n = array.GetLength(1);
            var ret = new float[n];

            for (int i = 0; i < n; i++)
                ret[i] = array[index, i];
            return ret;
        }
    }

    // Below copied from: https://github.com/migueldeicaza/TensorFlowSharp/blob/master/Learn/Datasets/MNIST.cs
    // Do NOT use this code as foundation for other things, quality is not great!

    // Stores the per-image MNIST information we loaded from disk 
    //
    // We store the data in two formats, byte array (as it came in from disk), and float array
    // where each 0..255 value has been mapped to 0.0f..1.0f
    public struct MnistImage
    {
        public int Cols, Rows;
        public byte[] Data;
        public float[] DataFloat;

        public MnistImage(int cols, int rows, byte[] data)
        {
            Cols = cols;
            Rows = rows;
            Data = data;
            DataFloat = new float[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                DataFloat[i] = Data[i] / 255f;
            }
        }
    }

    // Helper class used to load and work with the Mnist data set
    public class Mnist
    {
        // 
        // The loaded results
        //
        public MnistImage[] TrainImages, TestImages, ValidationImages;
        public byte[] TrainLabels, TestLabels, ValidationLabels;
        public byte[,] OneHotTrainLabels, OneHotTestLabels, OneHotValidationLabels;

        public BatchReader GetTrainReader() => new BatchReader(TrainImages, TrainLabels, OneHotTrainLabels);
        public BatchReader GetTestReader() => new BatchReader(TestImages, TestLabels, OneHotTestLabels);
        public BatchReader GetValidationReader() => new BatchReader(ValidationImages, ValidationLabels, OneHotValidationLabels);

        public class BatchReader
        {
            int start = 0;
            MnistImage[] source;
            byte[] labels;
            byte[,] oneHotLabels;

            internal BatchReader(MnistImage[] source, byte[] labels, byte[,] oneHotLabels)
            {
                this.source = source;
                this.labels = labels;
                this.oneHotLabels = oneHotLabels;
            }

            public (float[,], float[,]) NextBatch(int batchSize)
            {
                // TODO: Remove consts and allocs...
                var imageData = new float[batchSize, 784];
                var labelData = new float[batchSize, 10];

                int p = 0;
                for (int item = 0; item < batchSize; item++)
                {
                    Buffer.BlockCopy(source[start + item].DataFloat, 0, imageData, p, 784 * sizeof(float));
                    p += 784 * sizeof(float);
                    for (var j = 0; j < 10; j++)
                        labelData[item, j] = oneHotLabels[item + start, j];
                }

                start += batchSize;
                return (imageData, labelData);
            }
        }

        int Read32(Stream s)
        {
            var x = new byte[4];
            s.Read(x, 0, 4);
            var bigEndian = BitConverter.ToInt32(x, 0);
            return BigEndianToInt32(bigEndian);// DataConverter.BigEndian.GetInt32(x, 0);
        }

        int BigEndianToInt32(int bigEndian)
        {
            if (BitConverter.IsLittleEndian)
            {
                return (int)SwapBytes((uint)bigEndian);
            }
            return bigEndian;
        }

        public ushort SwapBytes(ushort x)
        {
            return (ushort)((ushort)((x & 0xff) << 8) | ((x >> 8) & 0xff));
        }

        public uint SwapBytes(uint x)
        {
            return ((x & 0x000000ff) << 24) +
                   ((x & 0x0000ff00) << 8) +
                   ((x & 0x00ff0000) >> 8) +
                   ((x & 0xff000000) >> 24);
        }

        public ulong SwapBytes(ulong x)
        {
            // swap adjacent 32-bit blocks
            x = (x >> 32) | (x << 32);
            // swap adjacent 16-bit blocks
            x = ((x & 0xFFFF0000FFFF0000) >> 16) | ((x & 0x0000FFFF0000FFFF) << 16);
            // swap adjacent 8-bit blocks
            return ((x & 0xFF00FF00FF00FF00) >> 8) | ((x & 0x00FF00FF00FF00FF) << 8);
        }

        MnistImage[] ExtractImages(Stream input, string file)
        {
            using (var gz = new GZipStream(input, CompressionMode.Decompress))
            {
                if (Read32(gz) != 2051)
                    throw new Exception("Invalid magic number found on the MNIST " + file);
                var count = Read32(gz);
                var rows = Read32(gz);
                var cols = Read32(gz);

                var result = new MnistImage[count];
                for (int i = 0; i < count; i++)
                {
                    var size = rows * cols;
                    var data = new byte[size];
                    gz.Read(data, 0, size);

                    result[i] = new MnistImage(cols, rows, data);
                }
                return result;
            }
        }


        byte[] ExtractLabels(Stream input, string file)
        {
            using (var gz = new GZipStream(input, CompressionMode.Decompress))
            {
                if (Read32(gz) != 2049)
                    throw new Exception("Invalid magic number found on the MNIST " + file);
                var count = Read32(gz);
                var labels = new byte[count];
                gz.Read(labels, 0, count);

                return labels;
            }
        }

        T[] Pick<T>(T[] source, int first, int last)
        {
            if (last == 0)
                last = source.Length;
            var count = last - first;
            var result = new T[count];
            Array.Copy(source, first, result, 0, count);
            return result;
        }

        // Turn the labels array that contains values 0..numClasses-1 into
        // a One-hot encoded array
        byte[,] OneHot(byte[] labels, int numClasses)
        {
            var oneHot = new byte[labels.Length, numClasses];
            for (int i = 0; i < labels.Length; i++)
            {
                oneHot[i, labels[i]] = 1;
            }
            return oneHot;
        }

        /// <summary>
        /// Reads the data sets.
        /// </summary>
        /// <param name="trainDir">Directory where the training data is downlaoded to.</param>
        /// <param name="numClasses">Number classes to use for one-hot encoding, or zero if this is not desired</param>
        /// <param name="validationSize">Validation size.</param>
        public void ReadDataSets(string trainDir, int numClasses = 10, int validationSize = 5000)
        {
            const string SourceUrl = "http://yann.lecun.com/exdb/mnist/";
            const string TrainImagesName = "train-images-idx3-ubyte.gz";
            const string TrainLabelsName = "train-labels-idx1-ubyte.gz";
            const string TestImagesName = "t10k-images-idx3-ubyte.gz";
            const string TestLabelsName = "t10k-labels-idx1-ubyte.gz";

            TrainImages = ExtractImages(Helper.MaybeDownload(SourceUrl, trainDir, TrainImagesName), TrainImagesName);
            TestImages = ExtractImages(Helper.MaybeDownload(SourceUrl, trainDir, TestImagesName), TestImagesName);
            TrainLabels = ExtractLabels(Helper.MaybeDownload(SourceUrl, trainDir, TrainLabelsName), TrainLabelsName);
            TestLabels = ExtractLabels(Helper.MaybeDownload(SourceUrl, trainDir, TestLabelsName), TestLabelsName);

            ValidationImages = Pick(TrainImages, 0, validationSize);
            ValidationLabels = Pick(TrainLabels, 0, validationSize);
            TrainImages = Pick(TrainImages, validationSize, 0);
            TrainLabels = Pick(TrainLabels, validationSize, 0);

            if (numClasses != -1)
            {
                OneHotTrainLabels = OneHot(TrainLabels, numClasses);
                OneHotValidationLabels = OneHot(ValidationLabels, numClasses);
                OneHotTestLabels = OneHot(TestLabels, numClasses);
            }
        }

        public static Mnist Load(string downloadPath)
        {
            var x = new Mnist();
            x.ReadDataSets(downloadPath);
            return x;
        }
    }

    public class Helper
    {
        public static Stream MaybeDownload(string urlBase, string trainDir, string file)
        {
            if (!Directory.Exists(trainDir))
                Directory.CreateDirectory(trainDir);
            var target = Path.Combine(trainDir, file);
            if (!File.Exists(target))
            {
                var wc = new WebClient();
                wc.DownloadFile(urlBase + file, target);
            }
            return File.OpenRead(target);
        }
    }
}
