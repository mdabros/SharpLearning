using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class GlobalizedBoundedNelderMeadOptimizerTest
    {
        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(-1)]
        [DataRow(null)]
        public void GlobalizedBoundedNelderMeadOptimizer_OptimizeBest(int? maxDegreeOfParallelism)
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };

            var sut = maxDegreeOfParallelism.HasValue ? 
                new GlobalizedBoundedNelderMeadOptimizer(parameters, 5, 1e-5, 10, 
                    maxDegreeOfParallelism: maxDegreeOfParallelism.Value) : 
                new GlobalizedBoundedNelderMeadOptimizer(parameters, 5, 1e-5, 10);

            var actual = sut.OptimizeBest(ObjectiveUtilities.Minimize);

            Assert.AreEqual(actual.Error, -0.99999949547279676, ObjectiveUtilities.Delta);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(actual.ParameterSet[0], -7.8547285710964134, ObjectiveUtilities.Delta);
            Assert.AreEqual(actual.ParameterSet[1], 6.2835515298977995, ObjectiveUtilities.Delta);
            Assert.AreEqual(actual.ParameterSet[2], -1.5851024386788885E-07, ObjectiveUtilities.Delta);
        }

        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(-1)]
        [DataRow(null)]
        public void GlobalizedBoundedNelderMeadOptimizer_Optimize(int? maxDegreeOfParallelism)
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };

            var sut = maxDegreeOfParallelism.HasValue ? 
                new GlobalizedBoundedNelderMeadOptimizer(parameters, 5, 1e-5, 10, 
                    maxDegreeOfParallelism: maxDegreeOfParallelism.Value) : 
                new GlobalizedBoundedNelderMeadOptimizer(parameters, 5, 1e-5, 10);

            var results = sut.Optimize(ObjectiveUtilities.MinimizeWeightFromHeight);
            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 37.71314535727786 }, 109.34381396310141),
                new OptimizerResult(new double[] { 37.7131485180996 }, 109.34381396350526)
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(expected.First().ParameterSet.First(), 
                actual.First().ParameterSet.First(), ObjectiveUtilities.Delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(), 
                actual.Last().ParameterSet.First(), ObjectiveUtilities.Delta);
        }
    }
}
