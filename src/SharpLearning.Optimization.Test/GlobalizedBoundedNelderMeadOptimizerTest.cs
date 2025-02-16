using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test;

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
            new(-10.0, 10.0, Transform.Linear),
            new(-10.0, 10.0, Transform.Linear),
            new(-10.0, 10.0, Transform.Linear),
        };

        var sut = CreateSut(maxDegreeOfParallelism, parameters);

        var actual = sut.OptimizeBest(Minimize);

        Assert.AreEqual(expected: -0.99592339271458108, actual.Error, Delta);
        Assert.AreEqual(expected: 3, actual.ParameterSet.Length);

        Assert.AreEqual(expected: 7.9170034654971069, actual.ParameterSet[0], Delta);
        Assert.AreEqual(expected: -3.1348067994029782, actual.ParameterSet[1], Delta);
        Assert.AreEqual(expected: -0.0020768773583485015, actual.ParameterSet[2], Delta);
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
            new(0.0, 100.0, Transform.Linear)
        };

        var sut = CreateSut(maxDegreeOfParallelism, parameters);

        var results = sut.Optimize(MinimizeWeightFromHeight);
        var actual = new OptimizerResult[] { results[0], results[^1] };

        var expected = new OptimizerResult[]
        {
            new([37.71323726440562], 109.34381430968727),
            new([37.713289997817874], 109.34381396345546),
        };

        Assert.AreEqual(expected[0].Error, actual[0].Error, Delta);
        Assert.AreEqual(expected[0].ParameterSet[0],
            actual[0].ParameterSet[0], Delta);

        Assert.AreEqual(expected[^1].Error, actual[^1].Error, Delta);
        Assert.AreEqual(expected[^1].ParameterSet[0],
            actual[^1].ParameterSet[0], Delta);
    }

    static GlobalizedBoundedNelderMeadOptimizer CreateSut(
        int? maybeMaxDegreeOfParallelism,
        MinMaxParameterSpec[] parameters)
    {
        const int defaultMaxDegreeOfParallelism = -1;

        var maxDegreeOfParallelism = maybeMaxDegreeOfParallelism ?? defaultMaxDegreeOfParallelism;

        var sut = new GlobalizedBoundedNelderMeadOptimizer(parameters,
            maxRestarts: 50,
            noImprovementThreshold: 1e-1,
            maxIterationsWithoutImprovement: 10,
            maxIterationsPrRestart: 0,
            maxFunctionEvaluations: 0,
            alpha: 1,
            gamma: 2,
            rho: -0.5,
            sigma: 0.5,
            seed: 324,
            maxDegreeOfParallelism: maxDegreeOfParallelism);

        return sut;
    }
}
