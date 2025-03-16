using System.Collections.Generic;
using System.Linq;
using BenchmarkDotNet.Attributes;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Benchmarks;

public static partial class Benchmarks
{
    [MemoryDiagnoser]
    public class RegressionModels
    {
        readonly IReadOnlyDictionary<string, ILearner<double>> m_learners =
            DefaultLearners.LearnerNameToLearnerRegression;
        readonly Dictionary<string, IPredictorModel<double>> m_models = [];

        F64Matrix m_features;
        double[] m_targets;

        [GlobalSetup]
        public void GlobalSetup()
        {
            (m_features, m_targets) = DataGenerator.GenerateRegressionData();
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
            model.Predict(m_features);
        }

        public IReadOnlyList<string> GetModels() =>
            m_models.Keys.ToArray();
    }
}
