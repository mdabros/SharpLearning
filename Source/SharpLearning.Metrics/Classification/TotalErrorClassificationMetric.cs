using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Classification
{
    /// <summary>
    /// Calculates the total error metric on a multi label or binary classification problem
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class TotalErrorClassificationMetric<T> : IClassificationMetric<T>
    {
        //readonly ClassificationMatrixStringConverter m_converter = new ClassificationMatrixStringConverter();
        //readonly ClassificationMatrix m_classificationMatrix = new ClassificationMatrix();

        /// <summary>
        /// Calculates the total error metric on a multi label or binary classification problem
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Error(T[] targets, T[] predictions)
        {
            var uniques = UniqueTargets(targets, predictions);

            var confusionMatrix = ClassificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = ClassificationMatrix.ErrorMatrix(uniques, confusionMatrix);

            return TotalError(uniques, confusionMatrix);
        }

        double TotalError(List<T> uniques, int[][] confusionMatrix)
        {
            var totalSum = confusionMatrix.SelectMany(v => v).Sum();
            var errorSum = totalSum;

            for (int row = 0; row < uniques.Count; ++row)
            {
                errorSum -= confusionMatrix[row][row];
            }

            return (double)errorSum / (double)totalSum;
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
            var error = TotalError(uniques, confusionMatrix);

            return ClassificationMatrixStringConverter.Convert(uniques, confusionMatrix, errorMatrix, error);
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
            var uniques = UniqueTargets(targets, predictions);

            var confusionMatrix = ClassificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = ClassificationMatrix.ErrorMatrix(uniques, confusionMatrix);
            var error = Error(targets, predictions);

            return ClassificationMatrixStringConverter.Convert(uniques, targetStringMapping, confusionMatrix, errorMatrix, error);
        }
    }
}
