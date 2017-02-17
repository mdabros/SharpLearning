using SharpLearning.Common.Interfaces;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// Metrics for calculating the error on continously valued predictions
    /// </summary>
    public interface IRegressionMetric : IMetric<double, double>
    {
        /// <summary>
        /// Metrics for calculating the error on continously valued predictions
        /// </summary>
        /// <param name="target"></param>
        /// <param name="predicted"></param>
        /// <returns></returns>
        new double Error(double[] target, double[] predicted);
    }
}
