
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
    }
}
