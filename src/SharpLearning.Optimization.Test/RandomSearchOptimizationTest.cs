using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test;

[TestClass]
public class RandomSearchOptimizerTest
{
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(-1)]
    [DataRow(null)]
    public void RandomSearchOptimizer_OptimizeBest(int? maxDegreeOfParallelism)
    {
        var parameters = new MinMaxParameterSpec[]
        {
            new(0.0, 100.0, Transform.Linear)
        };

        var sut = maxDegreeOfParallelism.HasValue ?
            new RandomSearchOptimizer(parameters, 100, 42, true, maxDegreeOfParallelism.Value) :
            new RandomSearchOptimizer(parameters, 100);

        var actual = sut.OptimizeBest(MinimizeWeightFromHeight);

        Assert.AreEqual(110.67173923600831, actual.Error, Delta);
        Assert.AreEqual(37.533294194160632, actual.ParameterSet.Single(), Delta);
    }

    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(-1)]
    [DataRow(null)]
    public void RandomSearchOptimizer_Optimize(int? maxDegreeOfParallelism)
    {
        var parameters = new MinMaxParameterSpec[]
        {
            new(10.0, 37.5, Transform.Linear)
        };

        var sut = maxDegreeOfParallelism.HasValue ?
            new RandomSearchOptimizer(parameters, 100, 42, true, maxDegreeOfParallelism.Value) :
            new RandomSearchOptimizer(parameters, 100);

        var actual = sut.Optimize(MinimizeWeightFromHeight);

        var expected = new OptimizerResult[]
        {
            new([28.3729278125674],  3690.81119818742),
            new([19.1529422843144], 14251.396910816733),
        };

        Assert.AreEqual(expected[0].Error, actual[0].Error, Delta);
        Assert.AreEqual(expected[0].ParameterSet[0],
            actual[0].ParameterSet[0], Delta);

        Assert.AreEqual(expected.Last().Error, actual.Last().Error, Delta);
        Assert.AreEqual(expected.Last().ParameterSet[0],
            actual.Last().ParameterSet[0], Delta);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void RandomSearchOptimizer_ArgumentCheck_ParameterRanges()
    {
        var sut = new RandomSearchOptimizer(null, 10);
    }
}
