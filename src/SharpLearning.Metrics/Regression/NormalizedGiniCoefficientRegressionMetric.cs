using System;
using System.Linq;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// Calculates the normalized gini coefficient
    /// https://en.wikipedia.org/wiki/Gini_coefficient
    /// </summary>
    public sealed class NormalizedGiniCoefficientRegressionMetric : IRegressionMetric
    {
        /// <summary>
        /// Calculates the normalized gini coefficient
        /// https://en.wikipedia.org/wiki/Gini_coefficient
        /// </summary>
        /// <param name="target"></param>
        /// <param name="predicted"></param>
        /// <returns></returns>
        public double Error(double[] target, double[] predicted)
        {
            return 1.0 - GiniCoefficient(target, predicted) / GiniCoefficient(target, target);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="target"></param>
        /// <param name="predicted"></param>
        /// <returns></returns>
        double GiniCoefficient(double[] target, double[] predicted)
        {
                if (target.Length != predicted.Length) 
                { throw new ArgumentException(); }
 
             var all = predicted.Zip(target, (prediction, actual) => new
                {
                    actualValue = actual,
                    predictedValue = prediction
                })
                .Zip(Enumerable.Range(1, target.Length), (ap, i) => new
                {
                    ap.actualValue,
                    ap.predictedValue,
                    originalIndex = i
                })
                .OrderByDescending(ap => ap.predictedValue) // important to sort descending by prediction
                .ThenBy(ap => ap.originalIndex); // secondary sorts to ensure unambiguous orders
 
             var totalActualLosses = target.Sum();

             double populationDelta = 1.0 / (double)target.Length;
             double accumulatedPopulationPercentageSum = 0;
             double accumulatedLossPercentageSum = 0;
 
             double giniSum = 0.0;
 
             foreach (var currentPair in all) 
             {
                 accumulatedLossPercentageSum += (currentPair.actualValue / totalActualLosses);
                 accumulatedPopulationPercentageSum += populationDelta;
                 giniSum += accumulatedLossPercentageSum - accumulatedPopulationPercentageSum;
             }
 
             var gini = giniSum / (double)target.Length;
             return gini;
        }
    }
}
