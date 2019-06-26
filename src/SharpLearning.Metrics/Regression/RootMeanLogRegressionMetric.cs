using System;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// Calculates the root mean logarithmic error between the targets and predictions e = Sum(Log(t +1) - log(p +1)))/length(t) 
    /// </summary>
    public sealed class RootMeanLogRegressionMetric : IRegressionMetric
    {
        /// <summary>
        /// Calculates the root mean logarithmic error between the targets and predictions e = Sum(Log(t +1) - log(p +1)))/length(t) 
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
            for (int i = 0; i < targets.Length; ++i)
            {
                var targetValue = targets[i];
                var estimate = predictions[i];
                var error = Math.Log(estimate + 1) - Math.Log(targetValue + 1);
                meanSquareError += error * error;
            }
            meanSquareError *= (1.0 / targets.Length);

            return Math.Sqrt(meanSquareError);
        }
    }
}
