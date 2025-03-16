using System.Collections.Generic;
using System.Linq;
using BenchmarkDotNet.Attributes;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Benchmarks;

public static partial class Benchmarks
{
    [MemoryDiagnoser]
    public class RegressionLearnersPredict
    {
        readonly IReadOnlyDictionary<string, ILearner<double>> m_learners =
            DefaultLearners.NameToRegressionLearner;

        // Data size for benchmarks.
        const int Rows = 1000;
        const int Cols = 10;
        F64Matrix m_features;
        double[] m_targets;
        Dictionary<string, IPredictorModel<double>> m_models;

        [GlobalSetup]
        public void GlobalSetup()
        {
            var seed = 42;
            m_targets = DataGenerator.GenerateDoubles(Rows, cols: 1, seed);
            var features = DataGenerator.GenerateDoubles(Rows, Cols, seed);
            m_features = new F64Matrix(features, Rows, Cols);

            m_models = new Dictionary<string, IPredictorModel<double>>();
            foreach (var (name, learner) in m_learners)
            {
                m_models[name] = learner.Learn(m_features, m_targets);
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
