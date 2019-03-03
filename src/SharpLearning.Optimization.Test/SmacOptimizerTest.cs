using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

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

            var actual = sut.OptimizeBest(MinimizeWeightFromHeight);

            Assert.AreEqual(109.616853578648, actual.Error, Delta);
            Assert.AreEqual(37.6315924979893, actual.ParameterSet.Single(), Delta);
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

            var actual = sut.OptimizeBest(Minimize);

            Assert.AreEqual(-0.964878416222769, actual.Error, Delta);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(-7.8487638560350819, actual.ParameterSet[0], Delta);
            Assert.AreEqual(6.2840940040927826, actual.ParameterSet[1], Delta);
            Assert.AreEqual(0.036385473812179825, actual.ParameterSet[2], Delta);
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

            var actual = sut.Optimize(MinimizeWeightFromHeight);

            var expected = new OptimizerResult[]
            {
                new OptimizerResult(new double[] { 90.513222660177 }, 114559.431919558),
                new OptimizerResult(new double[] { 41.8333740634068 },  806.274612132759),
            };

            Assert.AreEqual(expected.First().Error, actual.First().Error, Delta);
            Assert.AreEqual(expected.First().ParameterSet.First(), actual.First().ParameterSet.First(), Delta);

            Assert.AreEqual(expected.Last().Error, actual.Last().Error, Delta);
            Assert.AreEqual(expected.Last().ParameterSet.First(), actual.Last().ParameterSet.First(), Delta);
        }

        [TestMethod]
        public void SmacOptimizer_OptimizeBest_MultipleParameters_Open_Loop()
        {
            var results = new List<OptimizerResult>();

            OptimizerResult actual = RunOpenLoopOptimizationTest(results);

            Assert.AreEqual(-0.964878416222769, actual.Error, Delta);
            Assert.AreEqual(actual.ParameterSet.Length, 3);

            Assert.AreEqual(-7.8487638560350819, actual.ParameterSet[0], Delta);
            Assert.AreEqual(6.2840940040927826, actual.ParameterSet[1], Delta);
            Assert.AreEqual(0.036385473812179825, actual.ParameterSet[2], Delta);
        }

        OptimizerResult RunOpenLoopOptimizationTest(List<OptimizerResult> results)
        {
            var parameters = new MinMaxParameterSpec[]
            {
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
                new MinMaxParameterSpec(-10.0, 10.0, Transform.Linear),
            };

            var iterations = 80;
            var randomStartingPointsCount = 20;
            var functionEvaluationsPerIterationCount = 1;

            var sut = new SmacOptimizer(parameters,
                iterations: iterations,
                randomStartingPointCount: randomStartingPointsCount,
                functionEvaluationsPerIterationCount: functionEvaluationsPerIterationCount,
                localSearchPointCount: 10,
                randomSearchPointCount: 1000,
                seed: 42);

            var initialParameterSets = sut.ProposeParameterSets(randomStartingPointsCount, results);

            var initializationResults = sut.RunParameterSets(Minimize, initialParameterSets);
            results.AddRange(initializationResults);

            for (int i = 0; i < iterations; i++)
            {
                var parameterSets = sut.ProposeParameterSets(functionEvaluationsPerIterationCount, results);
                var iterationResults = sut.RunParameterSets(Minimize, parameterSets);
                results.AddRange(iterationResults);
            }

            var actual = results.Where(v => !double.IsNaN(v.Error)).OrderBy(r => r.Error).First(); ;
            return actual;
        }
    }
}
