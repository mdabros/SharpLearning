using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.DecisionTrees.LeafFactories
{
    /// <summary>
    /// Uses majority vote for leaf value calculation.
    /// Uses class frequencies adjusted using Laplace correction for probability estimates.
    /// This is improves probability estimates when using a single decision tree
    /// </summary>
    public sealed class LaplaceAdjustedClassificationLeafFactory : ILeafFactory
    {
        Dictionary<double, int> m_dictionary = new Dictionary<double, int>();

        /// <summary>
        /// Provides a classification leaf given a range of values. Using majority vote
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="values"></param>
        /// <param name="uniqueValues"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Create(IBinaryDecisionNode parent, double[] values, double[] uniqueValues)
        {
            var groups = values.GroupBy(v => v);
            var list = groups.OrderByDescending(g => g.Count()).ToList();
            var adjustedLength = values.Length + uniqueValues.Length;
            var probabilityFactor = 1.0 / adjustedLength;
            var probabilities = groups.ToDictionary(g => g.Key, g => (g.Count() + 1) * probabilityFactor);

            foreach (var unique in uniqueValues)
            {
                if(!probabilities.ContainsKey(unique))
                {
                    probabilities.Add(unique, 1 * probabilityFactor);
                }
            }

            var leafValue = list.First().Key;
            var leaf = new ClassificationBinaryDecisionNode(probabilities);
            leaf.Parent = parent;
            leaf.FeatureIndex = -1;
            leaf.Value = leafValue;
            return leaf;
        }

        /// <summary>
        /// Provides a classification leaf given a range of values and a calculation interval. using majority vote
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="values"></param>
        /// <param name="uniqueValues"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Create(IBinaryDecisionNode parent, double[] values, double[] uniqueValues, Interval1D interval)
        {
            m_dictionary.Clear();
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                var value = values[i];
                if (m_dictionary.ContainsKey(value))
                {
                    m_dictionary[value]++;
                }
                else
                {
                    m_dictionary.Add(value, 1);
                }
            }

            var leafValue = m_dictionary.OrderByDescending(kvp => kvp.Value).First().Key;

            var adjustedLength = interval.Length + uniqueValues.Length;
            var probabilityFactor = 1.0 / adjustedLength;
            var probabilities = m_dictionary.ToDictionary(g => g.Key, g => (g.Value + 1) * probabilityFactor);

            foreach (var unique in uniqueValues)
            {
                if (!probabilities.ContainsKey(unique))
                {
                    probabilities.Add(unique, 1 * probabilityFactor);
                }
            }

            var leaf = new ClassificationBinaryDecisionNode(probabilities);
            leaf.Parent = parent;
            leaf.FeatureIndex = -1;
            leaf.Value = leafValue;
            return leaf;
        }
    }
}
