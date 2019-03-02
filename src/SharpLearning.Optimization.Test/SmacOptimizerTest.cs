using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class SmacOptimizerTest
    {
        [TestMethod]
        public void SmacOptimizer_OptimizeBest()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };

            var sut = new SmacOptimizer(parameters);

            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(110.67173923600831, actual.Error, 0.00001);
            Assert.AreEqual(37.533294194160632, actual.ParameterSet.Single(), 0.00001);
        }

        [TestMethod]
        public void SmacOptimizer_OptimizeBest2()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };

            var sut = new SmacOptimizer(parameters);
            var actual = sut.OptimizeBest(Minimize2);

            Assert.AreEqual(actual.Error, -0.99999949547279676, 0.0000001);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(actual.ParameterSet[0], -7.8547285710964134, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[1], 6.2835515298977995, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[2], -1.5851024386788885E-07, 0.0000001);
        }

        [TestMethod]
        public void SmacOptimizer_Optimize()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(10.0, 37.5, Transform.Linear)
            };

            var sut = new RandomSearchOptimizer(parameters, 100);

            var actual = sut.Optimize(Minimize);

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 13.8749507052707 }, 23438.2157641635),
                new OptimizerResult(new double[] { 28.3729278125674 },  3690.81119818742),
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, 0.0001);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), 0.0001);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, 0.0001);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), 0.0001);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RandomSearchOptimizer_ArgumentCheck_ParameterRanges()
        {
            var sut = new RandomSearchOptimizer(null, 10);
        }

        OptimizerResult Minimize2(double[] x)
        {
            return new OptimizerResult(x, Math.Sin(x[0]) * Math.Cos(x[1]) * (1.0 / (Math.Abs(x[2]) + 1)));
        }

        OptimizerResult Minimize(double[] parameters)
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
