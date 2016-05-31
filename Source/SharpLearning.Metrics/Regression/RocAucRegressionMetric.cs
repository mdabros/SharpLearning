using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Regression
{
    /// <summary>
    /// The metric needs to know which target value is considered the positive
    /// </summary>
    public sealed class RocAucRegressionMetric : IRegressionMetric
    {
        readonly double m_positiveTarget;
        Dictionary<double, double> m_targetsToBinaryMapping;

        /// <summary>
        /// The metric needs to know which target value is considered the positive
        /// </summary>
        /// <param name="positiveTarget"></param>
        public RocAucRegressionMetric(double positiveTarget)
        {
            m_positiveTarget = positiveTarget;
        }

        /// <summary>
        /// The metric needs to know which target value is considered the positive.
        /// If targets are not already binary a mapping must be provided.
        /// </summary>
        /// <param name="positiveTarget"></param>
        /// <param name="targetsToBinaryMapping"></param>
        public RocAucRegressionMetric(double positiveTarget, Dictionary<double, double> targetsToBinaryMapping)
            : this(positiveTarget)
        {
            if(targetsToBinaryMapping.Values.Distinct().Count() != 2)
            { throw new ArgumentException("targetsToBinaryMapping must only map targets to two distinct values"); }
            m_targetsToBinaryMapping = targetsToBinaryMapping;
        }

        /// <summary>
        /// Calculates the roc auc error. That is 1.0 - Auc.
        /// </summary>
        /// <param name="targets">Target values</param>
        /// <param name="positiveTargetProbabilities">Probability estimates for the positive target value</param>
        /// <returns></returns>
        public double Error(double[] targets, double[] positiveTargetProbabilities)
        {
            if (targets.Distinct().Count() > 2 && m_targetsToBinaryMapping == null)
            { throw new ArgumentException("AucRegressionMetric only supports binary targets problems"); }

            if(m_targetsToBinaryMapping == null)
            {
                m_targetsToBinaryMapping = targets.Distinct().ToDictionary(t => t, t => t);
            }

            var targetProbabilities = targets.Zip(positiveTargetProbabilities, (t, p) => new { target = m_targetsToBinaryMapping[t], Probability = p });
            targetProbabilities = targetProbabilities.OrderByDescending(tp => tp.Probability);

            var counts = targetProbabilities.GroupBy(l => l.target).Select(l => new { Label = l.Key, Count = l.Count() });
                
            int negativeCount = counts.Where(s => !s.Label.Equals(m_positiveTarget)).Select(s => s.Count).Sum();;
            int positivesCount = counts.Where(s => s.Label.Equals(m_positiveTarget)).Select(s => s.Count).Sum();

            double auc = 0;
            double previousProbability = int.MinValue;
            long fpCount = 0, tpCount = 0, previousFpCount = 0, previousTpCount = 0;

            foreach (var targetProbability in targetProbabilities)
            {
                var probability = targetProbability.Probability;
                var target = targetProbability.target;

                if (probability != previousProbability)
                {
                    auc = auc +  trapezoidArea(fpCount * 1.0 / negativeCount, previousFpCount * 1.0 / negativeCount, 
                                               tpCount * 1.0 / positivesCount, previousTpCount * 1.0 / positivesCount);

                    previousProbability = probability;
                    previousFpCount = fpCount;
                    previousTpCount = tpCount;
                }
                if (target.Equals(m_positiveTarget))
                    tpCount = tpCount + 1;
                else
                    fpCount = fpCount + 1;
            }

            auc = auc + trapezoidArea(1.0, previousFpCount * 1.0 / negativeCount, 
                                      1.0, previousTpCount * 1.0 / positivesCount);
            return 1.0 - auc;
        }

        
        /// <summary>
        /// Caculate the trapezoidal area bound by the quad (X1,X2,Y1,Y2) 
        /// </summary>
        /// <param name="X1"></param>
        /// <param name="X2"></param>
        /// <param name="Y1"></param>
        /// <param name="Y2"></param>
        /// <returns></returns>
        double trapezoidArea(double X1, double X2, double Y1, double Y2)
        {
            double b = Math.Abs(X1 - X2);
            double height = (Y1 + Y2) / 2.0;
            return (b * height);
        }    
    }
}
