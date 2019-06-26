using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Classification
{
    /// <summary>
    /// 
    /// </summary>
    public static class ClassificationMatrix
    {
        /// <summary>
        /// Creates a confusion matrix from the provided targets and predictions
        /// </summary>
        /// <param name="uniqueTargets"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public static int[,] ConfusionMatrix<T>(List<T> uniqueTargets, T[] targets, T[] predictions)
        {
            var index = 0;
            var targetValueToTargetIndex = uniqueTargets.ToDictionary(t => t, t => index++);

            var targetPredictions = targets.Zip(predictions, (t, p) => new { Target = t, Prediction = p });
            var confusionMatrix = new int[uniqueTargets.Count, uniqueTargets.Count];

            foreach (var targetPrediction in targetPredictions)
            {
                var targetIndex = targetValueToTargetIndex[targetPrediction.Target];
                var predictionIndex = targetValueToTargetIndex[targetPrediction.Prediction];
                confusionMatrix[targetIndex, predictionIndex]++;
            }

            return confusionMatrix;
        }
        
        /// <summary>
        /// Creates an error matrix based on the provided confusion matrix
        /// </summary>
        /// <param name="uniqueTargets"></param>
        /// <param name="confusionMatrix"></param>
        /// <returns></returns>
        public static double[,] ErrorMatrix<T>(List<T> uniqueTargets, int[,] confusionMatrix)
        {
            var errorMatrix = new double[uniqueTargets.Count, uniqueTargets.Count];
            for (int row = 0; row < uniqueTargets.Count; ++row)
            {
                var rowSum = 0.0;
                for (int col = 0; col < uniqueTargets.Count; col++)
                {
                    rowSum += confusionMatrix[row, col];
                }

                for (int col = 0; col < uniqueTargets.Count; col++)
                {
                    double ratio = rowSum > 0.0 ? (confusionMatrix[row, col]) / rowSum : 0.0;
                    errorMatrix[row, col] = ratio;
                }
            }

            return errorMatrix;
        }
    }
}
