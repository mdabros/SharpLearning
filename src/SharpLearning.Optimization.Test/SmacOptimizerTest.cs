using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class SmacOptimizerTest
    {
        [TestMethod]
        public void SmacOptimizer_OptimizeBest_SingleParameter()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };

            var sut = new SmacOptimizer(parameters,
                iterations: 80,
                randomStartingPointCount: 20,
                functionEvaluationsPerIterationCount: 1,
                localSearchPointCount: 10,
                randomSearchPointCount: 1000,
                seed: 42);

            var actual = sut.OptimizeBest(ObjectiveUtilities.MinimizeWeightFromHeight);

            Assert.AreEqual(109.616853578648, actual.Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(37.6315924979893, actual.ParameterSet.Single(), ObjectiveUtilities.Delta);
        }

        [TestMethod]
        public void SmacOptimizer_OptimizeBest_MultipleParameters()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };

            var sut = new SmacOptimizer(parameters,
                iterations: 80,
                randomStartingPointCount: 20,
                functionEvaluationsPerIterationCount: 1,
                localSearchPointCount: 10,
                randomSearchPointCount: 1000,
                seed: 42);

            var actual = sut.OptimizeBest(ObjectiveUtilities.Minimize);

            Assert.AreEqual(-0.964878416222769, actual.Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(-7.8487638560350819, actual.ParameterSet[0], ObjectiveUtilities.Delta);
            Assert.AreEqual(6.2840940040927826, actual.ParameterSet[1], ObjectiveUtilities.Delta);
            Assert.AreEqual(0.036385473812179825, actual.ParameterSet[2], ObjectiveUtilities.Delta);
        }

        [TestMethod]
        public void SmacOptimizer_Optimize()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };

            var sut = new SmacOptimizer(parameters,
                iterations: 80,
                randomStartingPointCount: 20,
                functionEvaluationsPerIterationCount: 1,
                localSearchPointCount: 10,
                randomSearchPointCount: 1000,
                seed: 42);

            var actual = sut.Optimize(ObjectiveUtilities.MinimizeWeightFromHeight);

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 90.513222660177 }, 114559.431919558),
                new OptimizerResult(new double[] { 41.8333740634068 },  806.274612132759),
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), ObjectiveUtilities.Delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), ObjectiveUtilities.Delta);
        }
    }
}
