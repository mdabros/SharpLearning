
using SharpLearning.Containers.Views;

namespace SharpLearning.Metrics.Impurity
{
    /// <summary>
    /// Interface for impurity metrics
    /// </summary>
    public interface IImpurityMetric
    {
        /// <summary>
        /// Calculates the entropy of a sample
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        double Impurity(double[] values);

        /// <summary>
        /// Calculates the entropy of a sample within the provided interval
        /// </summary>
        /// <param name="values"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        double Impurity(double[] values, Interval1D interval);

        /// <summary>
        /// Calculates the weighted entropy within the provided interval
        /// </summary>
        /// <param name="values"></param>
        /// <param name="weights"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        double Impurity(double[] values, double[] weights, Interval1D interval);
    }
}
