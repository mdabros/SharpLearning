using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using SharpLearning.Optimization.OptimizerParameters;

namespace SharpLearning.Optimization.Test
{
    [TestClass]
    public class RandomSearchOptimizerTest
    {
        [TestMethod]
        public void RandomSearchOptimizer_OptimizeBest()
        {
            var parameters = new OptimizerParameter[] 
            {
                new OptimizerParameter(min: 0.0, max: 100.0, samplerScale: SamplerScale.Linear)
            };
            var sut = new RandomSearchOptimizer(parameters, 100);
            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(110.67173923600831, actual.Error, 0.00001);
            Assert.AreEqual(37.533294194160632, actual.ParameterSet.Single(), 0.00001);
        }

        [TestMethod]
        public void RandomSearchOptimizer_Optimize()
        {
            var parameters = new OptimizerParameter[] 
            {
                new OptimizerParameter(min: 10.0, max: 37.5, samplerScale: SamplerScale.Linear)
            };
            var sut = new RandomSearchOptimizer(parameters, 2);
            var actual = sut.Optimize(Minimize);

            var expected = new OptimizerResult[] 
            { 
              new OptimizerResult(new double[] { 28.372927812567415 }, 3690.8111981874217),
              new OptimizerResult(new double[] { 13.874950705270725 }, 23438.215764163542) 
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, 0.0001);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), 0.0001);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, 0.0001);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), 0.0001);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RandomSearchOptimizer_ArgumentCheck_ParameterRanges()
        {
            var sut = new RandomSearchOptimizer(null, 10);
        }

        OptimizerResult Minimize(double[] parameters)
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
