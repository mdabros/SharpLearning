using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// Calculaes the normalized gini coefficient
    /// https://en.wikipedia.org/wiki/Gini_coefficient
    /// </summary>
    public sealed class NormalizedGiniCoefficientRegressionMetric : IRegressionMetric
    {
        /// <summary>
        /// Calculaes the normalized gini coefficient
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
            var df = predicted.Zip(target, (p, t) => new { Prediction = p, Target = t }).ToArray();
            df = df.OrderByDescending(p => p.Prediction).ToArray();
            
            var orderedPredicted = df.Select(p => p.Prediction).ToArray();
            var orderedTargets = df.Select(p => p.Target).ToArray();

            var rand = new List<double>();
            for (int i = 0; i < df.Length; i++)
            {
                rand.Add(((double)i + 1.0) / (double)df.Length);
            }

            var totalPos = orderedTargets.Sum();
            var comPos = new List<double> { orderedTargets.First() };

            for (int i = 1; i < df.Length; i++)
            {
                comPos.Add(comPos[i - 1] + orderedTargets[i]);
            }

            var lorentz = new List<double>();
            foreach (var value in comPos)
            {
                lorentz.Add(value / totalPos);
            }

            var gini = new List<double>();
            for (int i = 0; i < df.Length; i++)
            {
                gini.Add(lorentz[i] - rand[i]);
            }

            var giniCoef = gini.Sum();

            return giniCoef;
        }
    }
}
