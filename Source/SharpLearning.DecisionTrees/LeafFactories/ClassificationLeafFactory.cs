using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.DecisionTrees.LeafFactories
{
    /// <summary>
    /// Uses majority vote for leaf value calculation
    /// </summary>
    public sealed class ClassificationLeafFactory : ILeafFactory
    {
        Dictionary<double, int> m_dictionary = new Dictionary<double, int>();

        /// <summary>
        /// Provides a classification leaf given a range of values. Using majority vote
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="values"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Create(IBinaryDecisionNode parent, double[] values)
        {
            var groups = values.GroupBy(v => v);
            var list = groups.OrderByDescending(g => g.Count()).ToList();

            var leafValue = list.First().Key;
            return new ContinousBinaryDecisionNode
            {
                Parent = parent,
                FeatureIndex = -1,
                Value = leafValue
            };
        }

        /// <summary>
        /// Provides a classification leaf given a range of values and a calculation interval. using majority vote
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="values"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Create(IBinaryDecisionNode parent, double[] values, Interval1D interval)
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
            return new ContinousBinaryDecisionNode
            {
                Parent = parent,
                FeatureIndex = -1,
                Value = leafValue
            };
        }
    }
}
