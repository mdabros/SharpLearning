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

            var sut = new SmacOptimizer(parameters, 80);

            var actual = sut.OptimizeBest(ObjectiveUtilities.MinimizeWeightFromHeight);

            Assert.AreEqual(109.42115405881532, actual.Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(37.669741473006894, actual.ParameterSet.Single(), ObjectiveUtilities.Delta);
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

            var sut = new SmacOptimizer(parameters, 80);
            var actual = sut.OptimizeBest(ObjectiveUtilities.Minimize);

            Assert.AreEqual(actual.Error, -0.98652950641642445, ObjectiveUtilities.Delta);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(actual.ParameterSet[0], -7.8353112367146238, ObjectiveUtilities.Delta);
            Assert.AreEqual(actual.ParameterSet[1], 6.2707440537729973, ObjectiveUtilities.Delta);
            Assert.AreEqual(actual.ParameterSet[2], 0.01339932438609992, ObjectiveUtilities.Delta);
        }

        [TestMethod]
        public void SmacOptimizer_Optimize()
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };

            var sut = new SmacOptimizer(parameters, 80);

            var actual = sut.Optimize(ObjectiveUtilities.MinimizeWeightFromHeight);

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 90.513222660177 }, 114559.431919558),
                new OptimizerResult(new double[] { 37.6697414730069 },  109.421154058815),
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), ObjectiveUtilities.Delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), ObjectiveUtilities.Delta);
        }
    }
}
