using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class BayesianOptimizerTest
    {
        [TestMethod]
        public void BayesianOptimizer_OptimizeBest()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };
            var sut = new BayesianOptimizer(parameters, 100, 5, 1);
            var actual = sut.OptimizeBest(ObjectiveUtilities.Minimize);

            Assert.AreEqual(actual.Error, -0.74765422244251278, 0.0001);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(-5.0065683270835173, actual.ParameterSet[0], ObjectiveUtilities.Delta);
            Assert.AreEqual(-9.67008227467075, actual.ParameterSet[1], ObjectiveUtilities.Delta);
            Assert.AreEqual(-0.24173704452893574, actual.ParameterSet[2], ObjectiveUtilities.Delta);
        }

        [TestMethod]
        public void BayesianOptimizer_Optimize()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };
            var sut = new BayesianOptimizer(parameters, 120, 5, 1);
            var results = sut.Optimize(ObjectiveUtilities.MinimizeWeightFromHeight);
            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 90.513222660177 }, 114559.431919558),
                new OptimizerResult(new double[] { 24.204380402436 },   7601.008090362)
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
