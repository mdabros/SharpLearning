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

            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(actual.Error, -0.99999949547279676, 0.0000001);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(actual.ParameterSet[0], -7.8547285710964134, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[1], 6.2835515298977995, 0.0000001);
            Assert.AreEqual(actual.ParameterSet[2], -1.5851024386788885E-07, 0.0000001);
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

            var results = sut.Optimize(Minimize2);
            var actual = new OptimizerResult[] { results.First(), results.Last() };

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 37.71314535727786 }, 109.34381396310141),
                new OptimizerResult(new double[] { 37.7131485180996 }, 109.34381396350526)
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, 0.0001);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), 0.0001);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, 0.0001);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), 0.0001);
        }

        OptimizerResult Minimize(double[] x)
        {
            return  new OptimizerResult(x, Math.Sin(x[0]) * Math.Cos(x[1]) * (1.0 / (Math.Abs(x[2]) + 1)));
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
