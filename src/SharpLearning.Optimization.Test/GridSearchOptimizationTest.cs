using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class GridSearchOptimizerTest
    {
        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(-1)]
        [DataRow(null)]
        public void GridSearchOptimizer_OptimizeBest(int? maxDegreeOfParallelism)
        {
            var parameters = new GridParameterSpec[] 
            {
                new GridParameterSpec(10.0, 20.0, 30.0, 35.0, 37.5, 40.0, 50.0, 60.0)
            };

            var sut = maxDegreeOfParallelism.HasValue ? 
                new GridSearchOptimizer(parameters, true, maxDegreeOfParallelism.Value) : 
                new GridSearchOptimizer(parameters);

            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(111.20889999999987, actual.Error, 0.00001);
            CollectionAssert.AreEqual(new double[] { 37.5 }, actual.ParameterSet);
        }

        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(-1)]
        [DataRow(null)]
        public void GridSearchOptimizer_Optimize(int? maxDegreeOfParallelism)
        {
            var parameters = new GridParameterSpec[] 
            {
                new GridParameterSpec(10.0, 37.5)
            };

            var sut = maxDegreeOfParallelism.HasValue ? 
                new GridSearchOptimizer(parameters, true, maxDegreeOfParallelism.Value) : 
                new GridSearchOptimizer(parameters);

            var actual = sut.Optimize(Minimize);

            var expected = new OptimizerResult[] 
            { 
              new OptimizerResult(new double[] { 37.5 }, 111.20889999999987),
              new OptimizerResult(new double[] { 10 }, 31638.9579) 
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, 0.0001);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), 0.0001);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, 0.0001);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), 0.0001);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GridSearchOptimizer_ArgumentCheck_ParameterRanges()
        {
            var sut = new GridSearchOptimizer(null, false);
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
