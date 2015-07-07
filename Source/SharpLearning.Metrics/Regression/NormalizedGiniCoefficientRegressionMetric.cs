using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
            return 1.0 - GiniCoefficient(predicted) / GiniCoefficient(target);
        }

        double GiniCoefficient(double[] values)
        {
            var ordered = values.OrderBy(v => v).ToArray();
            var height = 0.0;
            var area = 0.0;

            foreach (var value in ordered)
            {
                height += value;
                area += height - value / 2.0;
            }

            var fairArea = height * (double)values.Length / 2.0;

            return (fairArea - area) / fairArea;
        }
    }
}
