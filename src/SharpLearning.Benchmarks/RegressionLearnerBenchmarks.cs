using System;
using System.Collections.Generic;
using System.Linq;
using BenchmarkDotNet.Attributes;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.RandomForest.Learners;

namespace SharpLearning.Benchmarks;

[MemoryDiagnoser]
public class RegressionLearnerBenchmarks
{
    const int Rows = 1000;
    const int Cols = 10;
    F64Matrix m_features;
    double[] m_targets;

    [Params(
        typeof(RegressionDecisionTreeLearner),
        typeof(RegressionAdaBoostLearner),
        typeof(RegressionRandomForestLearner),
        typeof(RegressionExtremelyRandomizedTreesLearner),
        typeof(RegressionSquareLossGradientBoostLearner)
    )]
    public Type LearnerType;

    static readonly Dictionary<Type, Func<ILearner<double>>> LearnerFactory = new()
    {
        { typeof(RegressionDecisionTreeLearner), () => new RegressionDecisionTreeLearner() },
        { typeof(RegressionAdaBoostLearner), () => new RegressionAdaBoostLearner() },
        { typeof(RegressionRandomForestLearner), () => new RegressionRandomForestLearner() },
        { typeof(RegressionExtremelyRandomizedTreesLearner), () => new RegressionExtremelyRandomizedTreesLearner() },
        { typeof(RegressionSquareLossGradientBoostLearner), () => new RegressionSquareLossGradientBoostLearner() },
    };

    [GlobalSetup]
    public void GlobalSetup()
    {
        var seed = 42;
        m_targets = GenerateData(Rows, 1, seed);
        var features = GenerateData(Rows, Cols, seed);
        m_features = new F64Matrix(features, Rows, Cols);
    }

    [Benchmark]
    public void Learn()
    {
        var learner = LearnerFactory[LearnerType]();
        learner.Learn(m_features, m_targets);
    }

    static double[] GenerateData(int rows, int cols, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, rows * cols)
            .Select(i => random.NextDouble()).ToArray();
    }
}
