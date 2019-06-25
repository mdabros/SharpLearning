using System;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class RootMeanSquarePercentageRegressionMetric : IRegressionMetric
    {
        /// <summary>
        /// Calculates the root mean square percentage error between the targets and predictions e = Sqrt(sum((t - p / t)^2)/length(t))  
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(double[] targets, double[] predictions)
        {
            if (targets.Length != predictions.Length)
            {
                throw new ArgumentException("targets and predictions length do not match");
            }

            var meanSquareError = 0.0;
            var validEntries = 0;
            for (int i = 0; i < targets.Length; ++i)
            {
                var targetValue = targets[i];
                var estimate = predictions[i];

                if (targetValue == 0.0)
                {
                    continue;
                }

                validEntries++;
                var error = (targetValue - estimate) / targetValue;
                meanSquareError += error * error;
            }

            if (validEntries == 0)
            {
                return double.MaxValue;
            }

            meanSquareError *= (1.0 / validEntries);

            return Math.Sqrt(meanSquareError);
        }
    }
}
