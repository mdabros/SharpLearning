using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test;

[TestClass]
public class GlobalizedBoundedNelderMeadOptimizerTest
{
    [Ignore]
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

        Assert.AreEqual(actual.Error, -0.99999960731425908, Delta);
        Assert.AreEqual(actual.ParameterSet.Length, 3);

        Assert.AreEqual(actual.ParameterSet[0], -1.5711056814954487, Delta);
        Assert.AreEqual(actual.ParameterSet[1], -6.283490634742785, Delta);
        Assert.AreEqual(actual.ParameterSet[2], -2.9822323517533149E-07, Delta);
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
        var actual = new OptimizerResult[] { results.First(), results.Last() };

        var expected = new OptimizerResult[]
        {
            new([37.71314634450421], 109.3438139631394),
            new([37.713142445047254], 109.34381396345546)
        };

        Assert.AreEqual(expected.First().Error, actual.First().Error, Delta);
        Assert.AreEqual(expected.First().ParameterSet.First(),
            actual.First().ParameterSet.First(), Delta);

        Assert.AreEqual(expected.Last().Error, actual.Last().Error, Delta);
        Assert.AreEqual(expected.Last().ParameterSet.First(),
            actual.Last().ParameterSet.First(), Delta);
    }

    static GlobalizedBoundedNelderMeadOptimizer CreateSut(
        int? maybeMaxDegreeOfParallelism,
        MinMaxParameterSpec[] parameters)
    {
        const int DefaultMaxDegreeOfParallelism = -1;

        var maxDegreeOfParallelism = maybeMaxDegreeOfParallelism.HasValue ?
            maybeMaxDegreeOfParallelism.Value : DefaultMaxDegreeOfParallelism;

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
