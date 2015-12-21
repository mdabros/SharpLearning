using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class RandomSearchOptimizerTest
    {
        [TestMethod]
        public void RandomSearchOptimizer_Optimize()
        {
            var parameters = new double[][] { new double[]{ 0.0, 100.0 } };
            var sut = new RandomSearchOptimizer(parameters, 100);
            var actual = sut.Optimize(Minimize);

            Assert.AreEqual(110.67173923600831, actual.Error, 0.00001);
            Assert.AreEqual(37.533294194160632, actual.ParameterSet.Single(), 0.00001);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomSearchOptimizer_ArgumentCheck_MaxDegreeOfParallelism()
        {
            var parameters = new double[][] { new double[] { 10.0, 20.0, 30.0, 35.0, 37.5, 40.0, 50.0, 60.0 } };
            var sut = new RandomSearchOptimizer(parameters, 10, 42, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RandomSearchOptimizer_ArgumentCheck_ParameterRanges()
        {
            var sut = new RandomSearchOptimizer(null, 10);
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
