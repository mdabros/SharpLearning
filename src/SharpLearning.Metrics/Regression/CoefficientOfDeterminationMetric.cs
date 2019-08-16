using System;
using System.Linq;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class CoefficientOfDeterminationMetric : IRegressionMetric
    {
        /// <summary>
        /// Calculates coefficient of determination (r-squared) between the targets and predictions r-squared =  1 - sum((t-p)^2)/sum((t-t_mean)^2)
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

            var targetMean = targets.Sum() / targets.Length;
            var SStot = targets.Sum(target => Math.Pow(target - targetMean, 2));
            var SSres = 0.0;

            for (int i = 0; i < predictions.Length; i++)
            {
                SSres += Math.Pow(targets[i] - predictions[i], 2);
            }
                
            return SStot != 0.0?1 - SSres / SStot:0;
        }
    }
}
