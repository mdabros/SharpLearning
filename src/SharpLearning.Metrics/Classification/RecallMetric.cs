using System;
using System.Collections.Generic;

namespace SharpLearning.Metrics.Classification
{
    /// <summary>
    /// Calculates the recall metric (TP/(TP + FN)) on a binary classification problem
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class RecallMetric<T> : IClassificationMetric<T>
    {
        readonly T m_positiveTarget;

        /// <summary>
        /// Calculates the recall metric (TP/(TP + FN)) on a binary classification problem
        /// </summary>
        /// <param name="positiveTarget"></param>
        public RecallMetric(T positiveTarget)
        {
            if (positiveTarget == null) { throw new ArgumentNullException("positiveClassLabel"); }
            m_positiveTarget = positiveTarget;
        }

        /// <summary>
        /// Calculates the recall metric (TP/(TP + FN)) on a binary classification problem
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(T[] targets, T[] predictions)
        {
            var uniques = Utilities.UniqueTargetValues(targets, predictions);

            if (uniques.Count > 2)
            {
                throw new ArgumentException("RecallMetric only supports binary classification problems");
            }

            return 1.0 - Recall(targets, predictions);
        }

        double Recall(T[] targets, T[] predictions)
        {
            if (targets.Length != predictions.Length)
            {
                throw new ArgumentException("Predicted length differs from target length");
            }

            var truePositives = 0;
            var falseNegatives = 0;

            for (int i = 0; i < targets.Length; i++)
            {
                if(targets[i].Equals(m_positiveTarget) && predictions[i].Equals(m_positiveTarget))
                {
                    truePositives++;
                }
                else if (targets[i].Equals(m_positiveTarget) && !predictions[i].Equals(m_positiveTarget))
                {
                    falseNegatives++;
                }
            }

            if (truePositives + falseNegatives == 0)
            {
                return 0.0;
            }
            
            return (double)truePositives / ((double)truePositives + (double)falseNegatives);
        }

        /// <summary>
        /// Gets a string representation of the classification matrix with counts and percentages
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public string ErrorString(T[] targets, T[] predictions)
        {
            var error = Error(targets, predictions);
            return Utilities.ClassificationMatrixString(targets, predictions, error);
        }

        /// <summary>
        /// Gets a string representation of the classification matrix with counts and percentages
        /// Using the target names provided in the targetStringMapping
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="targetStringMapping"></param>
        /// <returns></returns>
        public string ErrorString(T[] targets, T[] predictions, Dictionary<T, string> targetStringMapping)
        {
            var error = Error(targets, predictions);
            return Utilities.ClassificationMatrixString(targets, predictions, error,
                targetStringMapping);
        }
    }
}
