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

        double TotalError(List<T> uniques, int[,] confusionMatrix)
        {
            var totalSum = Sum(confusionMatrix);
            var errorSum = totalSum;

            for (int row = 0; row < uniques.Count; ++row)
            {
                errorSum -= confusionMatrix[row,row];
            }

            return (double)errorSum / totalSum;
        }

        int Sum(int[,] confusionMatrix)
        {
            var rows = confusionMatrix.GetLength(0);
            var cols = confusionMatrix.GetLength(1);

            var sum = 0;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    sum += confusionMatrix[r, c];
                }
            }

            return sum;
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
