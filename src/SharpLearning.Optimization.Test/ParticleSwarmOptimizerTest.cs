using Microsoft.VisualStudio.TestTools.UnitTesting;
using static SharpLearning.Optimization.Test.ObjectiveUtilities;

namespace SharpLearning.Optimization.Test;

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
            new(-10.0, 10.0, Transform.Linear),
            new(-10.0, 10.0, Transform.Linear),
            new(-10.0, 10.0, Transform.Linear),
        };

        var sut = CreateSut(maxDegreeOfParallelism, parameters);

        var actual = sut.OptimizeBest(Minimize);

        Assert.AreEqual(-0.64324321766401094, actual.Error, Delta);
        Assert.AreEqual(3, actual.ParameterSet.Length);

        Assert.AreEqual(-4.92494268653156, actual.ParameterSet[0], Delta);
        Assert.AreEqual(10, actual.ParameterSet[1], Delta);
        Assert.AreEqual(-0.27508308116943514, actual.ParameterSet[2], Delta);
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
            new(0.0, 100.0, Transform.Linear)
        };

        var sut = CreateSut(maxDegreeOfParallelism, parameters);

        var results = sut.Optimize(MinimizeWeightFromHeight);

        var actual = new OptimizerResult[] { results[0], results[^1] };

        var expected = new OptimizerResult[]
        {
            new([38.1151505704492], 115.978346548015),
            new([37.2514904205637], 118.093289672808),
        };

        Assert.AreEqual(expected[0].Error, actual[0].Error, Delta);
        Assert.AreEqual(expected[0].ParameterSet[0],
            actual[0].ParameterSet[0], Delta);

        Assert.AreEqual(expected[^1].Error, actual[^1].Error, Delta);
        Assert.AreEqual(expected[^1].ParameterSet[0],
            actual[^1].ParameterSet[0], Delta);
    }

    static ParticleSwarmOptimizer CreateSut(
        int? maybeMaxDegreeOfParallelism,
        MinMaxParameterSpec[] parameters)
    {
        const int defaultMaxDegreeOfParallelism = -1;

        var maxDegreeOfParallelism = maybeMaxDegreeOfParallelism ?? defaultMaxDegreeOfParallelism;

        var sut = new ParticleSwarmOptimizer(parameters,
        maxIterations: 100,
        numberOfParticles: 10,
        c1: 2,
        c2: 2,
        seed: 42,
        maxDegreeOfParallelism: maxDegreeOfParallelism);

        return sut;
    }
}
