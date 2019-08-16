using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Classification
{
    internal static class Utilities
    {
        static internal List<T> UniqueTargetValues<T>(T[] targets, T[] predictions)
        {
            var uniquePredictions = predictions.Distinct();
            var uniqueTargets = targets.Distinct();
            var uniques = uniqueTargets.Union(uniquePredictions).ToList();

            uniques.Sort();
            return uniques;
        }

        internal static string ClassificationMatrixString<T>(T[] targets, T[] predictions, double error)
        {
            var uniques = UniqueTargetValues(targets, predictions);

            var confusionMatrix = ClassificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = ClassificationMatrix.ErrorMatrix(uniques, confusionMatrix);

            return ClassificationMatrixStringConverter.Convert(uniques, confusionMatrix, 
                errorMatrix, error);
        }

        internal static string ClassificationMatrixString<T>(T[] targets, T[] predictions, double error,
            Dictionary<T, string> targetStringMapping)
        {
            var uniques = UniqueTargetValues(targets, predictions);

            var confusionMatrix = ClassificationMatrix.ConfusionMatrix(uniques, targets, predictions);
            var errorMatrix = ClassificationMatrix.ErrorMatrix(uniques, confusionMatrix);

            return ClassificationMatrixStringConverter.Convert(uniques, targetStringMapping,
                confusionMatrix, errorMatrix, error);
        }
    }
}
