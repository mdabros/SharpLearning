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
    }
}
