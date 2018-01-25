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

        [TestMethod]
        public void MnistDeep()
        {
            const int imageSize = 28;
            const int featureCount = imageSize * imageSize;
            Assert.AreEqual(784, featureCount);
            const int classCount = 10;

            using (var g = new TFGraph())
            {
                TFOutput x = g.Placeholder(TFDataType.Float, new TFShape(-1, featureCount), "x");
                TFOutput expectedY = g.Placeholder(TFDataType.Float, new TFShape(-1, classCount), "y_");

                TFOutput W_zero = g.Const(new float[featureCount, classCount]);
                TFOutput b_zero = g.Const(new float[classCount]);

                TFOutput W = g.VariableV2(new TFShape(featureCount, classCount), TFDataType.Float, "W");
                // Only way to simply set zeros??
                TFOutput W_init = g.Assign(W, W_zero);

                TFOutput b = g.VariableV2(new TFShape(classCount), TFDataType.Float, "b");
                TFOutput b_init = g.Assign(b, b_zero);


                TFOutput m = g.MatMul(x, W, operName: "xW");
                TFOutput y = g.Add(m, b, operName: "y");

                // SoftmaxCrossEntropyWithLogits: gradient for this is not yet supported
                // see: https://github.com/tensorflow/tensorflow/pull/14727
                //var (perOutputLoss, backprop) = g.SoftmaxCrossEntropyWithLogits(y, expectedY, "softmax");
                TFOutput perOutputLoss = g.SquaredDifference(y, expectedY, "softmax");
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

                var mnist = Mnist.Load(DownloadPath);
                const int batchSize = 100;
                const int iterations = 200;

                using (var status = new TFStatus())
                using (var session = new TFSession(g))
                {
                    // Initialize variables
                    session.GetRunner().AddTarget(W_init.Operation).Run();
                    session.GetRunner().AddTarget(b_init.Operation).Run();

                    // Train (note that by using session.Run directly
                    //        we get a much more efficient loop, and it is easy to see what actually happens)
                    TFOutput[] inputs = new[] { x, expectedY };
                    TFOutput[] outputs = updates; // Gradient updates are currently the outputs ensuring these are applied
                    TFOperation[] targets = null; // TODO: It is possible to create a single operation target, instead of using outputs... how?
                    
                    TFBuffer runMetaData = null;
                    TFBuffer runOptions = null;
                    TFStatus trainStatus = new TFStatus();

                    var trainReader = mnist.GetTrainReader();
                    for (int i = 0; i < iterations; i++)
                    {
                        (float[,] inputBatch, float[,] labelBatch) = trainReader.NextBatch(batchSize);

                        TFTensor[] inputValues = new [] { new TFTensor(inputBatch), new TFTensor(labelBatch) };

                        TFTensor[] outputValues = session.Run(inputs, inputValues, outputs, 
                            targets, runMetaData, runOptions, trainStatus);
                    }

                    // Test trained model
                    TFOutput one = g.Const(new TFTensor(1));
                    TFOutput argMaxActual = g.ArgMax(y, one);
                    TFOutput argMaxExpected = g.ArgMax(expectedY, one);
                    TFOutput correctPrediction = g.Equal(argMaxActual, argMaxExpected);
                    TFOutput castCorrectPrediction = g.Cast(correctPrediction, TFDataType.Float);
                    TFOutput accuracy = g.ReduceMean(castCorrectPrediction);

                    var testReader = mnist.GetTestReader();
                    var (testImages, testLabels) = testReader.All();

                    TFTensor evaluatedAccuracy = session.GetRunner()
                        .AddInput(x, testImages)
                        .AddInput(expectedY, testLabels)
                        .Run(accuracy);

                    float acc = (float)evaluatedAccuracy.GetValue();

                    Log($"Accuracy {acc}");
                    Assert.AreEqual(acc, 0.797);
                }
            }
        }
    }
}
