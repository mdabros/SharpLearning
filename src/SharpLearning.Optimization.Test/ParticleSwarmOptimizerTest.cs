using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class ParticleSwarmOptimizerTest
    {
        [TestMethod]
        public void ParticleSwarmOptimizer_OptimizeBest()
        {
            var parameters = new ParameterBounds[]
            {
                new ParameterBounds(-10.0, 10.0, Transform.Linear),
                new ParameterBounds(-10.0, 10.0, Transform.Linear),
                new ParameterBounds(-10.0, 10.0, Transform.Linear),
            };
            var sut = new ParticleSwarmOptimizer(parameters, 100);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(actual.Error, -0.64324321766401094, 0.0000001);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(actual.ParameterSet[0], -4.92494268653156, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[1], 10, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[2], -0.27508308116943514, 0.0000001);
        }

        [TestMethod]
        public void ParticleSwarmOptimizer_Optimize()
        {
            var parameters = new ParameterBounds[]
            {
                new ParameterBounds(0.0, 100.0, Transform.Linear)
            };
            var sut = new ParticleSwarmOptimizer(parameters, 100);
            var results = sut.Optimize(Minimize2);
            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 37.660092259635064 }, 109.45936368750877),
                new OptimizerResult(new double[] { 39.038272502859328 }, 181.43166846962754)
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, 0.0001);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), 0.0001);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, 0.0001);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), 0.0001);
        }


        OptimizerResult Minimize(double[] x)
        {
            return new OptimizerResult(x, Math.Sin(x[0]) * Math.Cos(x[1]) * (1.0 / (Math.Abs(x[2]) + 1)));
        }

        OptimizerResult Minimize2(double[] parameters)
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
