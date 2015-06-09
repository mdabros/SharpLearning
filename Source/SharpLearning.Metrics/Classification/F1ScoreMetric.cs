using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Metrics.Classification
{
    public sealed class F1ScoreMetric<T> : IClassificationMetric<T>
    {
        readonly T m_positiveTarget;

        public F1ScoreMetric(T positiveTarget)
        {
            if (positiveTarget == null) { throw new ArgumentNullException("positiveClassLabel"); }
            m_positiveTarget = positiveTarget;
        }

        /// <summary>
        /// Calculates the F1Score metric 2 * precision * recall / (precision + recall) on a binary classification problem
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(T[] targets, T[] predictions)
        {
            var uniques = UniqueTargets(targets, predictions);

            if (uniques.Count > 2)
            { throw new ArgumentException("PrecisionMetric only supports binary classification problems"); }

            var precision = Precision(targets, predictions);
            var recall = Recall(targets, predictions);

            if (precision + recall == 0.0)
            {
                return 1.0 - 0.0;
            }

            var f1Score = 2 * precision * recall / (precision + recall);

            return 1.0 - f1Score;
        }

        double Precision(T[] targets, T[] predictions)
        {
            if (targets.Length != predictions.Length)
            { throw new ArgumentException("Predicted length differs from target length"); }

            var truePositives = 0;
            var falsePositves = 0;

            for (int i = 0; i < targets.Length; i++)
            {
                if (targets[i].Equals(m_positiveTarget) && predictions[i].Equals(m_positiveTarget))
                {
                    truePositives++;
                }
                else if (!targets[i].Equals(m_positiveTarget) && predictions[i].Equals(m_positiveTarget))
                {
                    falsePositves++;
                }
            }

            if (truePositives + falsePositves == 0)
            {
                return 0.0;
            }

            return (double)truePositives / ((double)truePositives + (double)falsePositves);
        }

        double Recall(T[] targets, T[] predictions)
        {
            if (targets.Length != predictions.Length)
            { throw new ArgumentException("Predicted length differs from target length"); }

            var truePositives = 0;
            var falseNegatives = 0;

            for (int i = 0; i < targets.Length; i++)
            {
                if (targets[i].Equals(m_positiveTarget) && predictions[i].Equals(m_positiveTarget))
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

        List<T> UniqueTargets(T[] targets, T[] predictions)
        {
            var uniquePredictions = predictions.Distinct();
            var uniqueTargets = targets.Distinct();
            var uniques = uniqueTargets.Union(uniquePredictions).ToList();

            uniques.Sort();
            return uniques;
        }

        /// <summary>
        /// Gets a string representation of the classification matrix with counts and percentages
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public string ErrorString(T[] targets, T[] predictions)
        {
            var uniques = UniqueTargets(targets, predictions);

            var confusionMatrix = ClassificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = ClassificationMatrix.ErrorMatrix(uniques, confusionMatrix);
            var error = Error(targets, predictions);

            return ClassificationMatrixStringConverter.Convert(uniques, confusionMatrix, errorMatrix, error);
        }
    }
}
