using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class SequentialModelBasedOptimizerTest
    {
        [TestMethod]
        public void SequentialModelBasedOptimizer_OptimizeBest()
        {
            var parameters = new double[][]
            {
                new double[] { -10.0, 10.0 },
                new double[] { -10.0, 10.0 },
                new double[] { -10.0, 10.0 }
            };

            var sut = new SequentialModelBasedOptimizer(parameters, 20, 5);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(actual.Error, -0.78008434504774338, 0.0000001);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(actual.ParameterSet[0], -7.9348252981893745, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[1], 0.44480908593386381, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[2], 0.15339395991115268, 0.0000001);
        }

        [TestMethod]
        public void SequentialModelBasedOptimizer_Optimize()
        {
            var parameters = new double[][] { new double[] { 0.0, 100.0 } };
            var sut = new SequentialModelBasedOptimizer(parameters, 20, 5);
            var results = sut.Optimize(Minimize2);
            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 37.713759626383109 }, 109.34382945231164),
                new OptimizerResult(new double[] { 66.810646591154239}, 34867.634010511123)
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, 0.0001);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), 0.0001);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, 0.0001);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), 0.0001);
        }


        OptimizerResult Minimize(double[] x)
        {
            return new OptimizerResult(x, Math.Sin(x[0]) * Math.Cos(x[1]) * (1.0 / (Math.Abs(x[2]) + 1)));
        }

        OptimizerResult Minimize2(double[] parameters)
        {
            var heights = new double[] { 1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83 };
            var weights = new double[] { 52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46 };

            var cost = 0.0;

            for (int i = 0; i < heights.Length; i++)
            {
                cost += (parameters[0] * heights[i] - weights[i]) * (parameters[0] * heights[i] - weights[i]);
            }

            return new OptimizerResult(parameters, cost);
        }
    }
}
