
namespace SharpLearning.Metrics.Entropy
{
    public sealed class GiniImpurityMetric : IEntropyMetric
    {
        readonly IntCustomDictionary m_dict = new IntCustomDictionary();

        public double Entropy(double[] values)
        {
            m_dict.Clear();

            for (int i = 0; i < values.Length; i++)
            {
                var targetInt = (int)values[i];

                int pos = m_dict.InitOrGetPosition(targetInt);
                int prevCount = m_dict.GetAtPosition(pos);
                m_dict.StoreAtPosition(pos, ++prevCount);
            }

            var totalInv = 1.0 / values.Length;
            var giniSum = 0.0;
            foreach (var pair in m_dict)
            {
                var ratio = pair.Value * totalInv;
                giniSum += ratio * ratio;
            }

            return 1 - giniSum;
        }
    }
}
