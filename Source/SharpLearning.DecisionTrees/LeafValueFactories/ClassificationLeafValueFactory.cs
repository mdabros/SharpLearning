using SharpLearning.Containers.Views;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.DecisionTrees.LeafValueFactories
{
    /// <summary>
    /// Uses majority vote for leaf value calculation
    /// </summary>
    public sealed class ClassificationLeafValueFactory : ILeafValueFactory
    {
        Dictionary<double, int> m_dictionary = new Dictionary<double, int>();

        /// <summary>
        /// Provides the value of a leaf given a range of values. Using majority vote
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public double Calculate(double[] values)
        {
            var groups = values.GroupBy(v => v);
            var list = groups.OrderByDescending(g => g.Count()).ToList();

            return list.First().Key;
        }

        /// <summary>
        /// Provides the value of a leaf given a range of values and a calculation interval. using majority vote
        /// </summary>
        /// <param name="values"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public double Calculate(double[] values, Interval1D interval)
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

            return m_dictionary.OrderByDescending(kvp => kvp.Value).First().Key;
        }
    }
}
