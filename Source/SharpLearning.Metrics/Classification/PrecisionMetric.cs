using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Metrics.Classification
{
    public sealed class PrecisionMetric<T> : IClassificationMetric<T>
    {
        readonly ClassificationMatrixStringConverter<T> m_converter = new ClassificationMatrixStringConverter<T>();
        readonly ClassificationMatrix<T> m_classificationMatrix = new ClassificationMatrix<T>();

        readonly T m_positiveTarget;

        public PrecisionMetric(T positiveTarget)
        {
            if (positiveTarget == null) { throw new ArgumentNullException("positiveClassLabel"); }
            m_positiveTarget = positiveTarget;
        }

        /// <summary>
        /// Calculates the precision metric (TP/(TP + FP)) on a binary classification problem
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(T[] targets, T[] predictions)
        {
            var uniques = UniqueTargets(targets, predictions);

            if (uniques.Count > 2)
            { throw new ArgumentException("PrecisionMetric only supports binary classification problems"); }

            return 1.0 - Precision(targets, predictions);
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

            if(truePositives + falsePositves == 0)
            {
                return 0.0;
            }

            return (double)truePositives / ((double)truePositives + (double)falsePositves);
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

            var confusionMatrix = m_classificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = m_classificationMatrix.ErrorMatrix(uniques, confusionMatrix);
            var error = 1.0 - Precision(targets, predictions);

            return m_converter.Convert(uniques, confusionMatrix, errorMatrix, error);
        }
    }
}
