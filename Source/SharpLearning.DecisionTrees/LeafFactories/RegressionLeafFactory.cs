using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;
using System.Linq;

namespace SharpLearning.DecisionTrees.LeafFactories
{
    /// <summary>
    /// Uses the mean for leaf value calculation
    /// </summary>
    public sealed class RegressionLeafFactory : ILeafFactory
    {
        /// <summary>
        /// Provides a regression leaf given a range of values and a calculation interval. Using the mean of the values
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="values"></param>
        /// <param name="uniqueValues"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Create(IBinaryDecisionNode parent, double[] values, double[] uniqueValues)
        {
            var leafValue = values.Sum() / values.Length;
            return new ContinousBinaryDecisionNode
            {
                Parent = parent,
                FeatureIndex = -1,
                Value = leafValue
            };
        }

        /// <summary>
        /// Provides a regression leaf given a range of values and a calculation interval. Using the mean of the values
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="values"></param>
        /// <param name="uniqueValues"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Create(IBinaryDecisionNode parent, double[] values, double[] uniqueValues, Interval1D interval)
        {
            var sum = 0.0;
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                sum += values[i];
            }
            var leafValue = sum / (double)interval.Length;
            
            return new ContinousBinaryDecisionNode
            {
                Parent = parent,
                FeatureIndex = -1,
                Value = leafValue
            };
        }
    }
}
