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
    public class ClassificationLearners
    {
        // Data size for benchmarks.
        const int Rows = 1000;
        const int Cols = 10;
        const int MinTargetValue = 0;
        const int MaxTargetValue = 10;
        F64Matrix m_features;
        double[] m_targets;

        // Define learners here. Use default parameters for benchmarks.
        readonly Dictionary<string, ILearner<double>> m_learners = new()
        {
            { nameof(ClassificationDecisionTreeLearner), new ClassificationDecisionTreeLearner() },
            { nameof(ClassificationAdaBoostLearner), new ClassificationAdaBoostLearner() },
            { nameof(ClassificationRandomForestLearner), new ClassificationRandomForestLearner() },
            { nameof(ClassificationExtremelyRandomizedTreesLearner), new ClassificationExtremelyRandomizedTreesLearner() },
            { nameof(ClassificationBinomialGradientBoostLearner), new ClassificationBinomialGradientBoostLearner() },
        };

        [GlobalSetup]
        public void GlobalSetup()
        {
            var seed = 42;
            m_targets = DataGenerator.GenerateIntegers(Rows, cols: 1,
                MinTargetValue, MaxTargetValue, seed);
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
