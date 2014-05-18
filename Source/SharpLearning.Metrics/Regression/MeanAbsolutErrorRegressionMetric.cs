using System;

namespace SharpLearning.Metrics.Regression
{
    public sealed class MeanAbsolutErrorRegressionMetric : IRegressionMetric
    {
        public double Error(double[] targets, double[] predicted)
        {
            var meanSquareError = 0.0;
            for (int i = 0; i < targets.Length; ++i)
            {
                var targetValue = targets[i];
                var estimate = predicted[i];
                var error = Math.Abs(targetValue - estimate);
                meanSquareError += error;
            }
            meanSquareError *= (1.0 / targets.Length);

            return meanSquareError;
        }
    }
}
