using BenchmarkDotNet.Attributes;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;

namespace SharpLearning.Benchmarks;

public static partial class Benchmarks
{
    [MemoryDiagnoser]
    public class ClassificationModels
    {
        const int Rows = 1000;
        const int Cols = 10;
        const int MinTargetValue = 0;
        const int MaxTargetValue = 10;
        F64Matrix m_features;
        double[] m_targets;

        // Define learners here. Use default parameters for benchmarks.
        readonly ClassificationDecisionTreeLearner m_classificationDecisionTreeLearner = new();
        ClassificationDecisionTreeModel m_classificationDecisionTreeModel;
        //readonly ClassificationAdaBoostLearner m_classificationAdaBoostLearner = new();
        //readonly ClassificationRandomForestLearner m_classificationRandomForestLearner = new();
        //readonly ClassificationExtremelyRandomizedTreesLearner m_classificationExtremelyRandomizedTreesLearner = new();
        //readonly ClassificationBinomialGradientBoostLearner m_classificationBinomialGradientBoostLearner = new();

        [GlobalSetup]
        public void GlobalSetup()
        {
            var seed = 42;
            m_targets = DataGenerator.GenerateIntegers(Rows, cols: 1,
                MinTargetValue, MaxTargetValue, seed);
            var features = DataGenerator.GenerateDoubles(Rows, Cols, seed);
            m_features = new F64Matrix(features, Rows, Cols);
            m_classificationDecisionTreeModel = m_classificationDecisionTreeLearner.Learn(m_features, m_targets);
        }

        [Benchmark]
        public void ClassificationDecisionTreeModel_Predict()
        {
            m_classificationDecisionTreeModel.Predict(m_features);
        }
    }
}
