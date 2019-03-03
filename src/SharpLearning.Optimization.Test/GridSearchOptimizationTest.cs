using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

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

            var actual = sut.OptimizeBest(MinimizeWeightFromHeight);

            Assert.AreEqual(111.20889999999987, actual.Error, Delta);
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

            var actual = sut.Optimize(MinimizeWeightFromHeight);

            var expected = new OptimizerResult[] 
            { 
              new OptimizerResult(new double[] { 37.5 }, 111.20889999999987),
              new OptimizerResult(new double[] { 10 }, 31638.9579) 
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, Delta);
            Assert.AreEqual(expected.First().ParameterSet.First(), 
                actual.First().ParameterSet.First(), Delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, Delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(),
                actual.Last().ParameterSet.First(), Delta);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void GridSearchOptimizer_ArgumentCheck_ParameterRanges()
        {
            var sut = new GridSearchOptimizer(null, false);
        }
    }
}
