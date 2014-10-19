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

        /// <summary>
        /// Calculates the precision metric (TP/(TP + FP)) on a multi label or binary classification problem
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(T[] targets, T[] predictions)
        {
            var uniques = UniqueTargets(targets, predictions);

            var confusionMatrix = m_classificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = m_classificationMatrix.ErrorMatrix(uniques, confusionMatrix);

            return 1.0 - Precision(uniques, confusionMatrix);
        }

        double Precision(List<T> uniques, int[][] confusionMatrix)
        {
            var totalSum = confusionMatrix.SelectMany(v => v).Sum();
            var falsePositives = totalSum;

            for (int row = 0; row < uniques.Count; ++row)
            {
                falsePositives -= confusionMatrix[row][row];
            }

            var truePositives = totalSum - falsePositives;

            return (double)truePositives / ((double)truePositives + (double)falsePositives);
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
            var error = Precision(uniques, confusionMatrix);

            return m_converter.Convert(uniques, confusionMatrix, errorMatrix, error);
        }
    }
}
