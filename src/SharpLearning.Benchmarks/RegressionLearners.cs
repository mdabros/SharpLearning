using System;
using System.Linq;
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
    public class RegressionLearners
    {
        const int Rows = 1000;
        const int Cols = 10;
        F64Matrix m_features;
        double[] m_targets;

        // Define learners here.
        readonly RegressionDecisionTreeLearner m_regressionDecisionTreeLearner = new();
        readonly RegressionAdaBoostLearner m_regressionAdaBoostLearner = new();
        readonly RegressionRandomForestLearner m_regressionRandomForestLearner = new();
        readonly RegressionExtremelyRandomizedTreesLearner m_regressionExtremelyRandomizedTreesLearner = new();
        readonly RegressionSquareLossGradientBoostLearner m_regressionSquareLossGradientBoostLearner = new();

        [GlobalSetup]
        public void GlobalSetup()
        {
            var seed = 42;
            m_targets = GenerateData(Rows, 1, seed);
            var features = GenerateData(Rows, Cols, seed);
            m_features = new F64Matrix(features, Rows, Cols);
        }

        [Benchmark]
        public void RegressionDecisionTreeLearner_Learn()
        {
            m_regressionDecisionTreeLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void RegressionAdaBoostLearner_Learn()
        {
            m_regressionAdaBoostLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void RegressionRandomForestLearner_Learn()
        {
            m_regressionRandomForestLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void RegressionExtremelyRandomizedTreesLearner_Learn()
        {
            m_regressionExtremelyRandomizedTreesLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void RegressionSquareLossGradientBoostLearner_Learn()
        {
            m_regressionSquareLossGradientBoostLearner.Learn(m_features, m_targets);
        }

        public static double[] GenerateData(int rows, int cols, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, rows * cols)
                .Select(i => random.NextDouble()).ToArray();
        }
    }
}
