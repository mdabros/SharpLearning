using System.Collections.Generic;

namespace SharpLearning.Metrics.Impurity
{
    /// <summary>
    /// Calculates the Gini impurity of a sample. Main use is for decision tree classification
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public sealed class GiniImpurityMetric : IImpurityMetric
    {
        readonly Dictionary<int, int> m_dictionary = new Dictionary<int, int>();

        /// <summary>
        /// Calculates the Gini impurity of a sample. Main use is for decision tree classification
        /// http://en.wikipedia.org/wiki/Decision_tree_learning
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public double Impurity(double[] values)
        {
            m_dictionary.Clear();

            for (int i = 0; i < values.Length; i++)
            {
                var targetKey = (int)values[i];

                if(!m_dictionary.ContainsKey(targetKey))
                {
                    m_dictionary.Add(targetKey, 1);
                }
                else
                {
                    m_dictionary[targetKey]++;
                }
            }

            var totalInv = 1.0 / (values.Length * values.Length);
            var giniSum = 0.0;

            foreach (var pair in m_dictionary)
            {
                giniSum += pair.Value * pair.Value;
            }

            giniSum = giniSum * totalInv;

            return 1 - giniSum;
        }
    }
}
