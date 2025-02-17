using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.RandomForest.Learners;

namespace SharpLearning.Benchmarks;

public static partial class Benchmarks
{
    [MemoryDiagnoser]
    public class ClassificationLearners
    {
        const int Rows = 1000;
        const int Cols = 10;
        F64Matrix m_features;
        double[] m_targets;

        // Define learners here.
        readonly ClassificationDecisionTreeLearner m_classificationDecisionTreeLearner = new();
        readonly ClassificationAdaBoostLearner m_classificationAdaBoostLearner = new();
        readonly ClassificationRandomForestLearner m_classificationRandomForestLearner = new();
        readonly ClassificationExtremelyRandomizedTreesLearner m_classificationExtremelyRandomizedTreesLearner = new();

        [GlobalSetup]
        public void GlobalSetup()
        {
            var seed = 42;
            m_targets = GenerateIntegers(Rows, 1, min: 0, max: 10, seed);
            var features = GenerateDoubles(Rows, Cols, seed);
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

        public static double[] GenerateDoubles(int rows, int cols, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, rows * cols)
                .Select(i => random.NextDouble()).ToArray();
        }

        public static double[] GenerateIntegers(int rows, int cols, int min, int max, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, rows * cols)
                .Select(i => random.Next(min, max)).Select(i => (double)i).ToArray();
        }
    }
}
