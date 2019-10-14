using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class BayesianOptimizerTest
    {

        private const int Seed = 42;
        private static Random Random = new Random(Seed);

        [TestMethod]
        public void BayesianOptimizer_OptimizeBest()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };
            var sut = new BayesianOptimizer(parameters, 100, 5, 1, maxDegreeOfParallelism: 1);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.74765422244251278, actual.Error, 0.0001);
            Assert.AreEqual(3, actual.ParameterSet.Length);

            Assert.AreEqual(-5.00656832708352, actual.ParameterSet[0], Delta);
            Assert.AreEqual(-9.67008227467075, actual.ParameterSet[1], Delta);
            Assert.AreEqual(-0.241737044528936, actual.ParameterSet[2], Delta);
        }

        [TestMethod]
        public void BayesianOptimizer_OptimizeBestInParallel()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };
            //high variance without fixed seed
            var sut = new BayesianOptimizer(parameters, 100, 5, 2, seed: Seed);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.736123479387088, actual.Error, Delta);
            Assert.AreEqual(3, actual.ParameterSet.Length);

            Assert.AreEqual(-4.37019084318084, actual.ParameterSet[0], Delta);
            Assert.AreEqual(-3.05224638108734, actual.ParameterSet[1], Delta);
            Assert.AreEqual(0.274598500819224, actual.ParameterSet[2], Delta);
        }

        [TestMethod]
        public void BayesianOptimizer_Optimize()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };
            var sut = new BayesianOptimizer(parameters, 120, 5, 1, maxDegreeOfParallelism: 1);
            var results = sut.Optimize(MinimizeWeightFromHeight);
            var actual = new OptimizerResult[] { results.First(), results.Last() }.OrderByDescending(o => o.Error);

            var expected = new OptimizerResult[]
                {
                    new OptimizerResult(new double[] { 90.513222660177 }, 114559.431919558),
                    new OptimizerResult(new double[] { 24.2043804024367 }, 7601.00809036235)
                };

            Assert.AreEqual(expected.First().Error, actual.First().Error, Delta);
            Assert.AreEqual(expected.First().ParameterSet.First(),
                actual.First().ParameterSet.First(), Delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, Delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(),
                actual.Last().ParameterSet.First(), Delta);
        }

        [TestMethod]
        public void BayesianOptimizer_OptimizeNonDeterministicInParallel()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0, 1, Transform.Linear, ParameterType.Discrete)
            };
            var sut = new BayesianOptimizer(parameters, iterations: 240, randomStartingPointCount: 5, functionEvaluationsPerIteration: 5,
                seed: Seed, maxDegreeOfParallelism: -1, allowMultipleEvaluations: true);
            var results = sut.Optimize(p => MinimizeNonDeterministic(p, Random));
            var actual = new OptimizerResult[] { results.First(), results.Last() }.OrderByDescending(o => o.Error);

            Assert.AreEqual(1, actual.First().Error);
            Assert.AreEqual(1, (int)actual.First().ParameterSet[0]);
        }

    }
}
