using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test;

[TestClass]
public class GridSearchOptimizerTest
{
    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(-1)]
    [DataRow(null)]
    public void GridSearchOptimizer_OptimizeBest(int? maxDegreeOfParallelism)
    {
        var parameters = new GridParameterSpec[]
        {
            new(10.0, 20.0, 30.0, 35.0, 37.5, 40.0, 50.0, 60.0)
        };

        var sut = maxDegreeOfParallelism.HasValue ?
            new GridSearchOptimizer(parameters, true, maxDegreeOfParallelism.Value) :
            new GridSearchOptimizer(parameters);

        var actual = sut.OptimizeBest(MinimizeWeightFromHeight);

        Assert.AreEqual(111.20889999999987, actual.Error, Delta);
        CollectionAssert.AreEqual(new double[] { 37.5 }, actual.ParameterSet);
    }

    [TestMethod]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(-1)]
    [DataRow(null)]
    public void GridSearchOptimizer_Optimize(int? maxDegreeOfParallelism)
    {
        var parameters = new GridParameterSpec[]
        {
            new(10.0, 20.0, 30.0, 35.0, 37.5, 40.0, 50.0, 60.0)
        };

        var sut = maxDegreeOfParallelism.HasValue ?
            new GridSearchOptimizer(parameters, true, maxDegreeOfParallelism.Value) :
            new GridSearchOptimizer(parameters);

        var actual = sut.Optimize(MinimizeWeightFromHeight);

        var expected = new OptimizerResult[]
        {
          new([10], 31638.9579),
          new([60], 20500.6279),
        };

        Assert.AreEqual(expected[0].Error, actual[0].Error, Delta);
        Assert.AreEqual(expected[0].ParameterSet[0],
            actual[0].ParameterSet[0], Delta);

        Assert.AreEqual(expected[^1].Error, actual[^1].Error, Delta);
        Assert.AreEqual(expected[^1].ParameterSet[0],
            actual[^1].ParameterSet[0], Delta);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void GridSearchOptimizer_ArgumentCheck_ParameterRanges()
    {
        var sut = new GridSearchOptimizer(null, false);
    }
}
