using System.Collections.Generic;
using System.Linq;
using BenchmarkDotNet.Attributes;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Benchmarks;

public static partial class Benchmarks
{
    [MemoryDiagnoser]
    public class ClassificationModels
    {
        readonly IReadOnlyDictionary<string, ILearner<double>> m_learners =
            DefaultLearners.LearnerNameToLearnerClassification;
        readonly Dictionary<string, IPredictorModel<double>> m_models = new();

        // For creating models.
        F64Matrix m_features;
        double[] m_targets;
        // For prediction.
        double[] m_featureRow;

        [GlobalSetup]
        public void GlobalSetup()
        {
            (m_features, m_targets) = DataGenerator.GenerateClassificationData();
            m_featureRow = m_features.Row(0);
            foreach (var (learnerName, learner) in m_learners)
            {
                var model = learner.Learn(m_features, m_targets);
                var modelName = DefaultLearners.LearnerNameToModelNameClassification[learnerName];
                m_models[modelName] = model;
            }
        }

        [Benchmark]
        [ArgumentsSource(nameof(GetModels))]
        public void Predict(string modelName)
        {
            var model = m_models[modelName];
            model.Predict(m_featureRow);
        }

        public IReadOnlyList<string> GetModels()
        {
            // Hack to ensure m_models is populated before call to GetModels.
            // This means `GlobalSetup` will be called twice.
            if (m_models.Count == 0)
            {
                GlobalSetup();
            }
            return m_models.Keys.ToArray();
        }
    }
}
