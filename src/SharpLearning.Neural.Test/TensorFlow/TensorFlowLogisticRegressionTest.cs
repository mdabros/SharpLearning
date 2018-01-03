using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorFlow;
using System.Diagnostics;

namespace SharpLearning.Neural.Test.TensorFlow
{
    [TestClass]
    public class TensorFlowLogisticRegressionTest
    {
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
                var param = new [] { W, b };

                var pred = g.Sigmoid(g.Add(g.Mul(X, W), b));

                // [WIP] Tensorflow c++ API still missing full gradient support.
                // BinaryCrossEntropy loss.
                var loss = g.ReduceMean(g.SigmoidCrossEntropyWithLogits(pred, Y));              
                var gradients = g.AddGradients(new [] {loss}, param);
                
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
    }
}
