using SharpLearning.Containers.Views;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using System.Collections.Generic;
using System.Diagnostics;

namespace SharpLearning.Metrics.Impurity
{
    public sealed class GiniImpurityMetric : IImpurityMetric
    {
        readonly IntCustomDictionary m_dict = new IntCustomDictionary();

        /// <summary>
        /// Calculates the Gini impurity of a sample. Main use is for decision tree classification
        /// http://en.wikipedia.org/wiki/Decision_tree_learning
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public double Impurity(double[] values)
        {
            m_dict.Clear();

            for (int i = 0; i < values.Length; i++)
            {
                var targetInt = (int)values[i];

                int pos = m_dict.InitOrGetPosition(targetInt);
                int prevCount = m_dict.GetAtPosition(pos);
                m_dict.StoreAtPosition(pos, ++prevCount);
            }

            var totalInv = 1.0 / (values.Length * values.Length);
            var giniSum = 0.0;

            foreach (var pair in m_dict)
            {
                giniSum += pair.Value * pair.Value;
            }

            giniSum = giniSum * totalInv;

            return 1 - giniSum;
        }

        /// <summary>
        /// Calculates the Gini impurity of a sample. Main use is for decision tree classification
        /// http://en.wikipedia.org/wiki/Decision_tree_learning
        /// </summary>
        /// <param name="values"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public double Impurity(double[] values, Interval1D interval)
        {
            m_dict.Clear();

            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                var targetInt = (int)values[i];

                int pos = m_dict.InitOrGetPosition(targetInt);
                int prevCount = m_dict.GetAtPosition(pos);
                m_dict.StoreAtPosition(pos, ++prevCount);
            }

            var totalInv = 1.0 / (interval.Length * interval.Length);
            var giniSum = 0.0;

            foreach (var pair in m_dict)
            {
                giniSum += pair.Value * pair.Value;
            }

            giniSum = giniSum * totalInv;

            return 1 - giniSum;
        }

        /// <summary>
        /// Calculates the weighted Gini impurity of a sample. Main use is for decision tree classification
        /// http://en.wikipedia.org/wiki/Decision_tree_learning
        /// </summary>
        /// <param name="values"></param>
        /// <param name="weights"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public double Impurity(double[] values, double[] weights, Interval1D interval)
        {
            m_dict.Clear();

            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                var targetInt = (int)values[i];

                int pos = m_dict.InitOrGetPosition(targetInt);
                int prevCount = m_dict.GetAtPosition(pos);
                m_dict.StoreAtPosition(pos, ++prevCount);
            }

            var weight = weights.Sum(interval);
            var totalInv = 1.0 / (weight * weight);
            var giniSum = 0.0;

            foreach (var pair in m_dict)
            {
                giniSum += pair.Value * pair.Value;
            }

            giniSum = giniSum * totalInv;

            return 1 - giniSum;
        }
    }
}
