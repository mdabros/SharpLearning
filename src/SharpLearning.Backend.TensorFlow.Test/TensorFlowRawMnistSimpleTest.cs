//#define USE_OLD_READER
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Backend.Testing;
using TensorFlow;

namespace SharpLearning.Backend.TensorFlow.Test
{
    [TestClass]
    public class TensorFlowRawMnistSimpleTest
    {
        const string DownloadPath = "MnistTest";
        static readonly Action<string> Log = t => { Trace.WriteLine(t); Console.WriteLine(t); };

        // https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/apply-gradient-descent
        // See https://github.com/tensorflow/tensorflow/pull/11377#issuecomment-324546254
        // https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc/gradients
        // https://github.com/tensorflow/tensorflow/pull/11377
        // See https://blog.manash.me/implementation-of-gradient-descent-in-tensorflow-using-tf-gradients-c111d783c78b

        [TestMethod]
        public void MnistSimple()
        {
            const int ImageSize = 28;
            const int FeatureCount = ImageSize * ImageSize;
            Assert.AreEqual(784, FeatureCount);
            const int ClassCount = 10;

            using (var g = new TFGraph())
            {
                TFOutput x = g.Placeholder(TFDataType.Float, new TFShape(-1, FeatureCount), "x");
                TFOutput y_ = g.Placeholder(TFDataType.Float, new TFShape(-1, ClassCount), "y_");

                TFOutput W_zero = g.Const(new float[FeatureCount, ClassCount]);
                TFOutput b_zero = g.Const(new float[ClassCount]);

                TFOutput W = g.VariableV2(new TFShape(FeatureCount, ClassCount), TFDataType.Float, "W");
                // Only way to simply set zeros??
                TFOutput W_init = g.Assign(W, W_zero);

                TFOutput b = g.VariableV2(new TFShape(ClassCount), TFDataType.Float, "b");
                TFOutput b_init = g.Assign(b, b_zero);


                TFOutput m = g.MatMul(x, W, operName: "xW");
                TFOutput y = g.Add(m, b, operName: "y");

                // SoftmaxCrossEntropyWithLogits: gradient for this is not yet supported
                // see: https://github.com/tensorflow/tensorflow/pull/14727
                //var (perOutputLoss, backprop) = g.SoftmaxCrossEntropyWithLogits(y, expectedY, "softmax");
                TFOutput perOutputLoss = g.SquaredDifference(y, y_, "softmax");
                TFOutput loss = g.ReduceMean(perOutputLoss, operName: "reducemean");

                TFOutput learningRate = g.Const(new TFTensor(0.01f));

                // TODO: How to get variables dynamically?
                TFOutput[] variables = new TFOutput[] { W, b };
                TFOutput[] gradients = g.AddGradients(new TFOutput[] { loss }, variables);


                TFOutput[] updates = new TFOutput[gradients.Length];
                for (int i = 0; i < gradients.Length; i++)
                {
                    updates[i] = g.ApplyGradientDescent(variables[i], learningRate, gradients[i]);
                }

                // Save graph definition
                using (var buffer = new TFBuffer())
                {
                    g.ToGraphDef(buffer);
                    var bytes = buffer.ToArray();
                    var dir = "../../../outputs/MnistSimple/";
                    if (!Directory.Exists(dir))
                    {
                        Directory.CreateDirectory(dir);
                    }
                    // This can't be used with TensorBoard, it is just the graph def
                    var filePath = dir + "graph.pb";
                    File.WriteAllBytes(filePath, bytes);
                }

#if USE_OLD_READER
                var mnist = Mnist.Load(DownloadPath);
#else
                var data = DataSets.Mnist.Load(DownloadPath);
#endif

                const int batchSize = 64;
                const int iterations = 200;

                var s = new Stopwatch();
                s.Start();

                using (var status = new TFStatus())
                using (var session = new TFSession(g))
                {
                    // Initialize variables
                    session.GetRunner().AddTarget(W_init.Operation).Run();
                    session.GetRunner().AddTarget(b_init.Operation).Run();

                    var initializeTime_ms = s.Elapsed.TotalMilliseconds;
                    s.Restart();

                    // Train (note that by using session.Run directly
                    //        we get a much more efficient loop, and it is easy to see what actually happens)
                    TFOutput[] inputs = new[] { x, y_ };
                    TFOutput[] outputs = updates; // Gradient updates are currently the outputs ensuring these are applied
                    TFOperation[] targets = null; // TODO: It is possible to create a single operation target, instead of using outputs... how?

                    TFBuffer runMetaData = null;
                    TFBuffer runOptions = null;
                    TFStatus trainStatus = new TFStatus();

#if USE_OLD_READER
                    var trainReader = mnist.GetTrainReader();
                    for (int i = 0; i < iterations; i++)
                    {
                        (float[,] inputBatch, float[,] labelBatch) = trainReader.NextBatch(batchSize);
                        TFTensor[] inputValues = new[] { new TFTensor(inputBatch), new TFTensor(labelBatch) };
#else
                    var rawTrainBatchEnumerator = data.CreateTrainBatchEnumerator(batchSize);
                    var trainBatchEnumerator = Convert(rawTrainBatchEnumerator, ClassCount);
                    for (int i = 0; i < iterations && trainBatchEnumerator.MoveNext(); i++)
                    {
                        var (inputBatch, labelBatch) = trainBatchEnumerator.CurrentBatch();
                        TFTensor[] inputValues = new[] {
                            TFTensor.FromBuffer(new TFShape(batchSize, FeatureCount), inputBatch, 0, inputBatch.Length),
                            TFTensor.FromBuffer(new TFShape(batchSize, ClassCount), labelBatch, 0, labelBatch.Length) };
#endif
                        TFTensor[] outputValues = session.Run(inputs, inputValues, outputs,
                            targets, runMetaData, runOptions, trainStatus);

                        trainStatus.Raise();
                    }

                    var trainTime_ms = s.Elapsed.TotalMilliseconds;
                    s.Restart();

                    // Test trained model
                    TFOutput one = g.Const(new TFTensor(1));
                    TFOutput argMaxActual = g.ArgMax(y, one);
                    TFOutput argMaxExpected = g.ArgMax(y_, one);
                    TFOutput correctPrediction = g.Equal(argMaxActual, argMaxExpected);
                    
                    TFOutput castCorrectPrediction = g.Cast(correctPrediction, TFDataType.Float);
                    TFOutput correctSum = g.ReduceSum(castCorrectPrediction);
                    TFOutput accuracy = g.ReduceMean(castCorrectPrediction);

#if USE_OLD_READER
                    var testReader = mnist.GetTestReader();
                    var (testImages, testLabels) = testReader.All();
                    TFTensor evaluatedAccuracy = session.GetRunner()
                        .AddInput(x, testImages)
                        .AddInput(y_, testLabels)
                        .Run(accuracy);
                    float acc = (float)evaluatedAccuracy.GetValue();
#else
                    //var rawTestBatchEnumerator = data.CreateTestBatchEnumerator(10000);
                    //var testBatchEnumerator = Convert(rawTestBatchEnumerator, classCount);
                    //testBatchEnumerator.MoveNext();
                    //var (testInputBatch, testLabelBatch) = testBatchEnumerator.CurrentBatch();

                    //TFTensor testInputBatchTensor = TFTensor.FromBuffer(new TFShape(10000, featureCount), testInputBatch, 0, testInputBatch.Length);
                    //TFTensor testLabelBatchTensor = TFTensor.FromBuffer(new TFShape(10000, classCount), testLabelBatch, 0, testLabelBatch.Length);

                    //TFTensor evaluatedCorrectPrediction = session.GetRunner()
                    //    .AddInput(x, testInputBatchTensor)
                    //    .AddInput(y_, testLabelBatchTensor)
                    //    .Run(correctPrediction);

                    //var ps = (bool[])evaluatedCorrectPrediction.GetValue();

                    //var correctCount = ps.Count(p => p);
                    //var incorrectCount = ps.Length - correctCount;
                    //var acc = correctCount / (float)ps.Length;

                    //TFTensor evaluatedAccuracy = session.GetRunner()
                    //    .AddInput(x, testInputBatchTensor)
                    //    .AddInput(y_, testLabelBatchTensor)
                    //    .Run(accuracy);
                    //float acc = (float)evaluatedAccuracy.GetValue();

                    // Batch evaluation is faster

                    var testBatchSize = 100;
                    Debug.Assert(data.TestTargets.Data.Length % testBatchSize == 0); // Or we need to do something that handles remaining samples
                    var rawTestBatchEnumerator = data.CreateTestBatchEnumerator(testBatchSize);
                    var testBatchEnumerator = Convert(rawTestBatchEnumerator, ClassCount);
                    var totalCount = 0;
                    var correctCount = 0;
                    while (testBatchEnumerator.MoveNext())
                    {
                        var (testInputBatch, testLabelBatch) = testBatchEnumerator.CurrentBatch();

                        TFTensor testInputBatchTensor = TFTensor.FromBuffer(new TFShape(testBatchSize, FeatureCount), testInputBatch, 0, testInputBatch.Length);
                        TFTensor testLabelBatchTensor = TFTensor.FromBuffer(new TFShape(testBatchSize, ClassCount), testLabelBatch, 0, testLabelBatch.Length);

                        TFTensor evaluatedCorrectPrediction = session.GetRunner()
                            .AddInput(x, testInputBatchTensor)
                            .AddInput(y_, testLabelBatchTensor)
                            .Run(correctPrediction);

                        var ps = (bool[])evaluatedCorrectPrediction.GetValue();

                        totalCount += ps.Length;
                        foreach (var p in ps)
                        {
                            if (p) { ++correctCount; }
                        }
                    }

                    var incorrectCount = totalCount - correctCount;
                    var acc = correctCount / (float)totalCount;
#endif
                    var testTime_ms = s.Elapsed.TotalMilliseconds;
                    s.Restart();

                    Log($"Accuracy {acc} Initialize {initializeTime_ms,6:F1} Train {trainTime_ms,6:F1} Test {testTime_ms,6:F1} ");
                    //Assert.AreEqual(0.7882000207901001, acc); // With validation set of 5000 of tests 10000
                    Assert.AreEqual(0.7839999794960022, acc);// With no validation set
                }
            }
        }

        private static IFlatBatchFeaturesTargetEnumerator<float, float> Convert(
            IFlatBatchFeaturesTargetEnumerator<byte, byte> rawTrainBatchEnumerator, int classCount)
        {
            var trainBatchEnumerator = rawTrainBatchEnumerator
                // TODO: We really want to do this before batch enumerator so same as for tests..
                //.From().To(fb => (float)fb, tb => (int)tb);
                //.Feature().To(fb => (float)fb)
                .Feature().To(fb => fb * (1.0f / byte.MaxValue))
                .Target().ToOneHot(t => t, 1.0f, classCount);
            return trainBatchEnumerator;
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
    }
}
