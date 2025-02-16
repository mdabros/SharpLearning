using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;

namespace SharpLearning.Benchmarks;

[MemoryDiagnoser]
public class DecisionTreeLearnerBenchmarks
{
    const int Rows = 1000;
    const int Cols = 10;
    F64Matrix m_features;
    double[] m_targets;

    ILearner<double> m_learner;

    [GlobalSetup]
    public void GlobalSetup()
    {
        var seed = 42;
        m_targets = GenerateData(Rows, 1, seed);
        var features = GenerateData(Rows, Cols, seed);
        m_features = new F64Matrix(features, Rows, Cols);

        // Use default parameters.
        m_learner = new RegressionDecisionTreeLearner();
    }

    [Benchmark]
    public void RegressionDecisionTreeLearner_Learn()
    {
        m_learner.Learn(m_features, m_targets);
    }

    public static double[] GenerateData(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, rows * cols)
            .Select(i => random.NextDouble()).ToArray();
    }
}
