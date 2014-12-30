using SharpLearning.Learners.Interfaces;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// Metrics for calculating the error on floating point predictions
    /// </summary>
    public interface IRegressionMetric : IMetric<double>
    {
        new double Error(double[] target, double[] predicted);
    }
}
