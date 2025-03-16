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

public static partial class Benchmarks
{
    [MemoryDiagnoser]
    public class RegressionLearners
    {
        // Data size for benchmarks.
        const int Rows = 1000;
        const int Cols = 10;
        F64Matrix m_features;
        double[] m_targets;

        // Define learners here. Use default parameters for benchmarks.
        readonly Dictionary<string, ILearner<double>> m_learners = new()
        {
            { nameof(RegressionDecisionTreeLearner), new RegressionDecisionTreeLearner() },
            { nameof(RegressionAdaBoostLearner), new RegressionAdaBoostLearner() },
            { nameof(RegressionRandomForestLearner), new RegressionRandomForestLearner() },
            { nameof(RegressionExtremelyRandomizedTreesLearner), new RegressionExtremelyRandomizedTreesLearner() },
            { nameof(RegressionAbsoluteLossGradientBoostLearner), new RegressionAbsoluteLossGradientBoostLearner() },
            { nameof(RegressionHuberLossGradientBoostLearner), new RegressionHuberLossGradientBoostLearner() },
            { nameof(RegressionQuantileLossGradientBoostLearner), new RegressionQuantileLossGradientBoostLearner() },
            { nameof(RegressionSquareLossGradientBoostLearner), new RegressionSquareLossGradientBoostLearner() }
        };

        [GlobalSetup]
        public void GlobalSetup()
        {
            var seed = 42;
            m_targets = DataGenerator.GenerateDoubles(Rows, cols: 1, seed);
            var features = DataGenerator.GenerateDoubles(Rows, Cols, seed);
            m_features = new F64Matrix(features, Rows, Cols);
        }

        [Benchmark]
        [ArgumentsSource(nameof(GetLearners))]
        public void Learn(string learnerName)
        {
            var learner = m_learners[learnerName];
            learner.Learn(m_features, m_targets);
        }

        public IReadOnlyList<string> GetLearners() =>
            m_learners.Keys.ToArray();
    }
}
