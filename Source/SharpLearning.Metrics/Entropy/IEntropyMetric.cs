
using SharpLearning.Containers.Views;
namespace SharpLearning.Metrics.Entropy
{
    public interface IEntropyMetric
    {
        /// <summary>
        /// Calculates the entropy of a sample
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        double Entropy(double[] values);

        /// <summary>
        /// Calculates the entropy of a sample within the provided interval
        /// </summary>
        /// <param name="values"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        double Entropy(double[] values, Interval1D interval);

        /// <summary>
        /// Calculates the weighted entropy within the provided interval
        /// </summary>
        /// <param name="values"></param>
        /// <param name="weights"></param>
        /// <param name="interval"></param>
        /// <returns></returns>
        double Entropy(double[] values, double[] weights, Interval1D interval);
    }
}
