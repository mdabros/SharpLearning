using SharpLearning.Containers.Views;
using System.Linq;

namespace SharpLearning.DecisionTrees.LeafValueFactories
{
    /// <summary>
    /// Uses the mean for leaf value calculation
    /// </summary>
    public sealed class RegressionLeafValueFactory : ILeafValueFactory
    {
        /// <summary>
        /// Provides the value of a leaf given a range of values. Using the mean of the range
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>

        public double Calculate(double[] values)
        {
            var mean = values.Sum() / values.Length;
            return mean;
        }

        /// <summary>
        /// Provides the value of a leaf given a range of values and a calculation interval. using the mean of the range
        /// </summary>
        /// <param name="values"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        public double Calculate(double[] values, Interval1D interval)
        {
            var sum = 0.0;
            for (int i = interval.FromInclusive; i < interval.ToExclusive; i++)
            {
                sum += values[i];
            }
            return sum / (double)interval.Length;
        }
    }
}
