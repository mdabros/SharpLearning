using BenchmarkDotNet.Attributes;
using SharpLearning.AdaBoost.Learners;
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
        const int Rows = 10000;
        const int Cols = 50;
        const int MinTargetValue = 0;
        const int MaxTargetValue = 10;
        F64Matrix m_features;
        double[] m_targets;

        // Define learners here. Use default parameters for benchmarks.
        readonly ClassificationDecisionTreeLearner m_classificationDecisionTreeLearner = new();
        readonly ClassificationAdaBoostLearner m_classificationAdaBoostLearner = new();
        readonly ClassificationRandomForestLearner m_classificationRandomForestLearner = new();
        readonly ClassificationExtremelyRandomizedTreesLearner m_classificationExtremelyRandomizedTreesLearner = new();
        readonly ClassificationBinomialGradientBoostLearner m_classificationBinomialGradientBoostLearner = new();

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
        public void ClassificationDecisionTreeLearner_Learn()
        {
            m_classificationDecisionTreeLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void ClassificationAdaBoostLearner_Learn()
        {
            m_classificationAdaBoostLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void ClassificationRandomForestLearner_Learn()
        {
            m_classificationRandomForestLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn()
        {
            m_classificationExtremelyRandomizedTreesLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void ClassificationBinomialGradientBoostLearner_Learn()
        {
            m_classificationBinomialGradientBoostLearner.Learn(m_features, m_targets);
        }
    }
}
