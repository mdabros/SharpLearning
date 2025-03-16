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
            DefaultLearners.NameToRegressionLearner;
        readonly Dictionary<string, IPredictorModel<double>> m_models = [];

        F64Matrix m_features;
        double[] m_targets;

        [GlobalSetup]
        public void GlobalSetup()
        {
            (m_features, m_targets) = DataGenerator.GenerateRegressionData();
            foreach (var (_, learner) in m_learners)
            {
                var model = learner.Learn(m_features, m_targets);
                m_models[model.GetType().Name] = model;
            }
        }

        [Benchmark]
        [ArgumentsSource(nameof(GetLearners))]
        public void Predict(string learnerName)
        {
            var model = m_models[learnerName];
            model.Predict(m_features);
        }

        public IReadOnlyList<string> GetLearners() =>
            m_learners.Keys.ToArray();
    }
}
