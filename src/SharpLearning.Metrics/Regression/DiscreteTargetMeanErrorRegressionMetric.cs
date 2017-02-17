using System;
using System.Linq;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// Zips the targets and predictions and groups by the values availible in targets.
    /// The resuling error is the mean error of each group. The internal metric used for each group
    /// Is provided as a constructor parameter.
    /// This metric only makes sence if the target values have a descrete nature. That is, not too many unique values.
    /// An example could be targets with the values: 0, 1, 2, 3, 4, 5.
    /// </summary>
    public sealed class DiscreteTargetMeanErrorRegressionMetric : IRegressionMetric
    {
        readonly IRegressionMetric m_regressionMetric;

        /// <summary>
        /// Zips the targets and predictions and groups by the values availible in targets.
        /// The resuling error is the mean error of each group. The internal metric used for each group
        /// Is provided as a constructor parameter.
        /// This metric only makes sence if the target values have a descrete nature. That is, not too many unique values.
        /// An example could be targets with the values: 0, 1, 2, 3, 4, 5.
        /// </summary>
        /// <param name="regressionMetric">Default is MeanSquaredError</param>
        public DiscreteTargetMeanErrorRegressionMetric(IRegressionMetric regressionMetric)
        {
            if (regressionMetric == null) { throw new ArgumentNullException("regressionMetric"); }
            m_regressionMetric = regressionMetric;
        }

        /// <summary>
        /// Zips the targets and predictions and groups by the values availible in targets.
        /// The resuling error is the mean error of each group. The internal metric used for each group
        /// Is provided as a constructor parameter.
        /// This metric only makes sence if the target values have a descrete nature. That is, not too many unique values.
        /// An example could be targets with the values: 0, 1, 2, 3, 4, 5.
        /// 
        /// Default internal metric is MeanSquaredErrorRegressionMetric.
        /// </summary>
        public DiscreteTargetMeanErrorRegressionMetric()
            : this(new MeanSquaredErrorRegressionMetric())
        {
        }

        /// <summary>
        /// The resuling error is the mean error of each group. The internal metric used for each group
        /// Is provided as a constructor parameter.
        /// This metric only makes sence if the target values have a descrete nature. That is, not too many unique values.
        /// An example could be targets with the values: 0, 1, 2, 3, 4, 5.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="predicted"></param>
        /// <returns></returns>
        public double Error(double[] target, double[] predicted)
        {
            var discreteGroups = target.Zip(predicted, (t, p) => new { Target = t, Predicted = p })
                .GroupBy(tp => tp.Target).ToArray();

            var error = 0.0;
            foreach (var group in discreteGroups)
            {
                var targets = group.Select(tp => tp.Target).ToArray();
                var predictions = group.Select(tp => tp.Predicted).ToArray();

                error += m_regressionMetric.Error(targets, predictions);
            }

            error /= (double)discreteGroups.Length;
            return error;
        }
    }
}
