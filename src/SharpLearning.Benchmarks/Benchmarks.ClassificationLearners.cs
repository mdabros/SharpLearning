using System.Collections.Generic;
using System.Linq;
using BenchmarkDotNet.Attributes;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Benchmarks;

public static partial class Benchmarks
{
    [MemoryDiagnoser]
    public class ClassificationLearners
    {
        readonly IReadOnlyDictionary<string, ILearner<double>> m_learners =
            DefaultLearners.NameToClassificationLearner;

        F64Matrix m_features;
        double[] m_targets;

        [GlobalSetup]
        public void GlobalSetup()
        {
            (m_features, m_targets) = DataGenerator.GenerateClassificationData();
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
