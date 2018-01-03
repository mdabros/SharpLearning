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
            Trace.WriteLine("Linear regression");
            // Parameters
            var learning_rate = 0.001f;
            var training_epochs = 100;
            var display_step = 5;

            // Training data
            var train_x = new float[] {
                3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f
            };
            var train_y = new float[] {
                1.7f,2.76f,2.09f,3.19f,1.694f,1.573f,3.366f,2.596f,2.53f,1.221f,
                 2.827f,3.465f,1.65f,2.904f,2.42f,2.94f,1.3f
            };

            var n_samples = train_x.Length;

            var dataType = TFDataType.Float;

            using (var g = new TFGraph())
            {
                var s = new TFSession(g);
                var rng = new Random(21);
                // tf Graph Input

                var X = g.Placeholder(dataType);
                var Y = g.Placeholder(dataType);

                var W = g.VariableV2(TFShape.Scalar, dataType, operName: "W");
                var initW = g.Assign(W, g.Const(0.1f));

                var b = g.VariableV2(TFShape.Scalar, dataType, operName: "b");
                var initb = g.Assign(b, g.Const(1f));
                var param = new [] { W, b };

                var pred = g.Add(g.Mul(X, W), b);

                // [WIP] Tensorflow c++ API still missing full gradient support.
                //var loss = g.Div(g.ReduceSum(g.Pow(g.Sub(pred, Y), g.Const(2f))), g.Mul(g.Const(2f), g.Const((float)n_samples)));               
                // using mean square loss instead.
                var loss = g.ReduceMean(g.Square(g.Sub(pred, Y)));              
                var gradients = g.AddGradients(new [] {loss}, param);
                
                // figure out how to do updates on lists of params and gradients.
                var updateW = g.Assign(W, g.Add(W, g.Mul(g.Const(learning_rate), gradients[0])));
                var updateB = g.Assign(b, g.Add(b, g.Mul(g.Const(learning_rate), gradients[1])));
                                                              
                // initialize variables
                s.GetRunner().AddTarget(initW.Operation).Run();
                s.GetRunner().AddTarget(initb.Operation).Run();

                var observations = train_x.Zip(train_y, (x, y) => new { X = x, Y = y }).ToArray();                
                for (int epoch = 0; epoch < training_epochs; epoch++)
                {
                    foreach (var observation in observations)
                    {

                        // run optimization loop.
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
