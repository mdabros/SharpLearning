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

        Assert.AreEqual(expected: -0.99999960731425908, actual.Error, Delta);
        Assert.AreEqual(expected: 3, actual.ParameterSet.Length);

        const double delta = 1e-3;
        Assert.AreEqual(expected: -1.5711056814954487, actual.ParameterSet[0], delta);
        Assert.AreEqual(expected: -6.283490634742785, actual.ParameterSet[1], delta);
        Assert.AreEqual(expected: -2.9822323517533149E-07, actual.ParameterSet[2], delta);
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
            new([37.71314634450421], 109.3438139631394),
            new([37.713142445047254], 109.34381396345546),
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
            noImprovementThreshold: 1e-5,
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
