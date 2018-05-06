using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class BayesianOptimizerTest
    {
        readonly double m_delta = 0.000001;

        [TestMethod]
        public void BayesianOptimizer_OptimizeBest()
        {
            var parameters = new ParameterBounds[]
            {
                new ParameterBounds(-10.0, 10.0, Transform.Linear),
                new ParameterBounds(-10.0, 10.0, Transform.Linear),
                new ParameterBounds(-10.0, 10.0, Transform.Linear),
            };
            var sut = new BayesianOptimizer(parameters, 100, 5, 1);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(actual.Error, -0.73736717818644282, 0.0001);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(actual.ParameterSet[0], 7.8124302242940438, m_delta);
            Assert.AreEqual(actual.ParameterSet[1], -3.2319937242343988, m_delta);
            Assert.AreEqual(actual.ParameterSet[2], 0.34947285910578074, m_delta);
        }

        [TestMethod]
        public void BayesianOptimizer_Optimize()
        {
            var parameters = new ParameterBounds[]
            {
                new ParameterBounds(0.0, 100.0, Transform.Linear)
            };
            var sut = new BayesianOptimizer(parameters, 120, 5, 1);
            var results = sut.Optimize(Minimize2);
            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 37.524597457388694 }, 110.80326835639002),
                new OptimizerResult(new double[] { 98.240981063917729 }, 150512.62292441679)
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, m_delta);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), m_delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, m_delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), m_delta);
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
