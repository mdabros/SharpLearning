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
            var sut = new BayesianOptimizer(parameters, 100, 5, 1, seed: 42, runParallel: false);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.76070603822760785, actual.Error, Delta);
            Assert.AreEqual(3, actual.ParameterSet.Length);

            Assert.AreEqual(1.6078245041928358, actual.ParameterSet[0], Delta);
            Assert.AreEqual(-8.9735394990879769, actual.ParameterSet[1], Delta);
            Assert.AreEqual(-0.18217921731163855, actual.ParameterSet[2], Delta);
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
            var sut = new BayesianOptimizer(parameters, 100, 5, 1, seed: 42, runParallel: true);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.76070603822760785, actual.Error, Delta);
            Assert.AreEqual(3, actual.ParameterSet.Length);

            Assert.AreEqual(1.6078245041928358, actual.ParameterSet[0], Delta);
            Assert.AreEqual(-8.9735394990879769, actual.ParameterSet[1], Delta);
            Assert.AreEqual(-0.18217921731163855, actual.ParameterSet[2], Delta);
        }

        [TestMethod]
        public void BayesianOptimizer_Optimize()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };
            var sut = new BayesianOptimizer(parameters, 120, 5, 1, seed: 42, runParallel: false);
            var results = sut.Optimize(MinimizeWeightFromHeight);
            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 43.216748276360683 }, 1352.8306605984087),
                new OptimizerResult(new double[] { 38.201425707992833 }, 119.1316225267316)
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, Delta);
            Assert.AreEqual(expected.First().ParameterSet.First(),
                actual.First().ParameterSet.First(), Delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, Delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(),
                actual.Last().ParameterSet.First(), Delta);
        }
    }
}
