using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class ParticleSwarmOptimizerTest
    {
        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(-1)]
        [DataRow(null)]
        public void ParticleSwarmOptimizer_OptimizeBest(int? maxDegreeOfParallelism)
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };

            var sut = maxDegreeOfParallelism.HasValue ? 
                new ParticleSwarmOptimizer(parameters, 100, maxDegreeOfParallelism: maxDegreeOfParallelism.Value) : 
                new ParticleSwarmOptimizer(parameters, 100);

            var actual = sut.OptimizeBest(ObjectiveUtilities.Minimize);

            Assert.AreEqual(-0.64324321766401094, actual.Error, ObjectiveUtilities.Delta);
            Assert.AreEqual(3, actual.ParameterSet.Length);

            Assert.AreEqual(-4.92494268653156, actual.ParameterSet[0], ObjectiveUtilities.Delta);
            Assert.AreEqual(10, actual.ParameterSet[1], ObjectiveUtilities.Delta);
            Assert.AreEqual(-0.27508308116943514, actual.ParameterSet[2], ObjectiveUtilities.Delta);
        }

        [TestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(-1)]
        [DataRow(null)]
        public void ParticleSwarmOptimizer_Optimize(int? maxDegreeOfParallelism)
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(0.0, 100.0, Transform.Linear)
            };

            var sut = maxDegreeOfParallelism.HasValue ? 
                new ParticleSwarmOptimizer(parameters, 100, maxDegreeOfParallelism: maxDegreeOfParallelism.Value) : 
                new ParticleSwarmOptimizer(parameters, 100);

            var results = sut.Optimize(ObjectiveUtilities.MinimizeWeightFromHeight);

            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 38.1151505704492 }, 115.978346548015),
                new OptimizerResult(new double[] { 37.2514904205637 }, 118.093289672808),
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
