using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorFlow;

namespace SharpLearning.Neural.Test.TensorFlow
{
    [TestClass]
    public class TensorFlowLogisticRegressionTest
    {
        [TestMethod]
        public void Run_TensorFlow_Logistic_Regression()
        {
            Console.WriteLine("Logistic regression");
            // Parameters
            var learning_rate = 0.01;
            var training_epochs = 1000;
            var display_step = 50;

            // Training data
            var train_x = new double[] {
                3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1
            };
            var train_y = new double[] {
                1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                 2.827,3.465,1.65,2.904,2.42,2.94,1.3
            };
            var n_samples = train_x.Length;

            using (var g = new TFGraph())
            {
                var s = new TFSession(g);
                var rng = new Random();
                // tf Graph Input

                var X = g.Placeholder(TFDataType.Float);
                var Y = g.Placeholder(TFDataType.Float);
                var W = g.Variable(g.Const(rng.Next()), operName: "weight");
                var b = g.Variable(g.Const(rng.Next()), operName: "bias");
                var pred = g.Add(g.Mul(X, W), b);

                var cost = g.Div(g.ReduceSum(g.Pow(g.Sub(pred, Y), g.Const(2))), g.Mul(g.Const(2), g.Const(n_samples)));

                // STuck here: TensorFlow bindings need to surface gradient support
                // waiting on Google for this
                // https://github.com/migueldeicaza/TensorFlowSharp/issues/25

                // work in progress.

                // optimizer is not supported yet. TODO: Add manauel SGD implementation to loop.
                //var optimizer = new GradientDescentOptimizer(learning_rate).Minimize(cost);

                //var init = g.GetGlobalVariablesInitializer();
                //s.GetRunner().Run(init);

                var observations = train_x.Zip(train_y, (x, y) => new { X = x, Y = y }).ToArray();
                var runner = s.GetRunner();
                for (int epoch = 0; epoch < training_epochs; epoch++)
                {
                    foreach (var observation in observations)
                    {
                        runner
                        .AddInput(X, observation.X)
                        .AddInput(Y, observation.Y);
                        //.Run(optimizer);
                    }

                    // Display logs per epoch step
                    if ((epoch + 1) % display_step == 0)
                    {
                        var c = runner
                            .AddInput(X, train_x)
                            .AddInput(Y, train_y)
                            .Run(cost);

                        Console.WriteLine($"Epoch: {epoch + 1}, cost={c}, W={runner.Run(W)}, b={runner.Run(b)}");
                    }
                }

                // end work in progress
            }
        }
    }
}
