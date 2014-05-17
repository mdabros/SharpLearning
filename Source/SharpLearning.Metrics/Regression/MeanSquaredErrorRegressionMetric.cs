
using System;
namespace SharpLearning.Metrics.Regression
{
    public class MeanSquaredErrorRegressionMetric : IRegressionMetric
    {
        /// <summary>
        /// Calculates the meansquared error between the targets and predictions e = (t - p)^2 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(double[] targets, double[] predictions)
        {
            if (targets.Length != predictions.Length) { throw new ArgumentException("targets and predictions length do not match"); }
            
            var meanSquareError = 0.0;
            for (int i = 0; i < targets.Length; ++i)
            {
                var targetValue = targets[i];
                var estimate = predictions[i];
                var error = targetValue - estimate;
                meanSquareError += error * error;
            }
            meanSquareError *= (1.0 / targets.Length);

            return meanSquareError;
        }
    }
}
