using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class GridSearchOptimizerTest
    {
        [TestMethod]
        public void GridSearchOptimizer_Optimize()
        {
            var parameters = new double[][] { new double[]{ 10.0, 20.0, 30.0, 35.0, 37.5, 40.0, 50.0, 60.0 } };
            var sut = new GridSearchOptimizer(parameters);
            var actual = sut.Optimize(Minimize);

            Assert.AreEqual(111.20889999999987, actual.Error, 0.00001);
            CollectionAssert.AreEqual(new double[] { 37.5 }, actual.ParameterSet);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void GridSearchOptimizer_ArgumentCheck_MaxDegreeOfParallelism()
        {
            var parameters = new double[][] { new double[] { 10.0, 20.0, 30.0, 35.0, 37.5, 40.0, 50.0, 60.0 } };
            var sut = new GridSearchOptimizer(parameters, 0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GridSearchOptimizer_ArgumentCheck_ParameterRanges()
        {
            var sut = new GridSearchOptimizer(null, 10);
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
