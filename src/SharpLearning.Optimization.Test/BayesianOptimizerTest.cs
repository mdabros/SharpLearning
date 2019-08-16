using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

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
            var sut = new BayesianOptimizer(parameters, 100, 5, 1, maxDegreeOfParallelism: 1);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.706891104009228, actual.Error, 0.0001);
            Assert.AreEqual(3, actual.ParameterSet.Length);

            Assert.AreEqual(1.48822282975923, actual.ParameterSet[0], Delta);
            Assert.AreEqual(-2.90695000109586, actual.ParameterSet[1], Delta);
            Assert.AreEqual(-0.371192242191729, actual.ParameterSet[2], Delta);
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
            //high variance on few iterations
            var sut = new BayesianOptimizer(parameters, 100, 5, 2);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.947532236193108, actual.Error, 1);
            Assert.AreEqual(3, actual.ParameterSet.Length);
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
                    new OptimizerResult(new double[] { 37.948484550206 },   111.6175259058)
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
                new MinMaxParameterSpec(0.0, 1.0, Transform.Linear)
            };
            var sut = new BayesianOptimizer(parameters, 100, 10, 5, allowMultipleEvaluations: true);
            var results = sut.Optimize(MinimizeNonDeterministic);
            var actual = new OptimizerResult[] { results.First(), results.Last() }.OrderByDescending(o => o.Error);

            Assert.AreEqual(2d, actual.First().Error);
            Assert.AreEqual(0.421397202844451, actual.First().ParameterSet.First(), 0.5);
        }

    }
}
