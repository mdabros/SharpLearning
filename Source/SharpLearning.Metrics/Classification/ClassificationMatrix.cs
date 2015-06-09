using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Classification
{
    public static class ClassificationMatrix
    {
        /// <summary>
        /// Creates a confusion matrix from the provided targets and predicitons
        /// </summary>
        /// <param name="uniqueTargets"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public static int[][] ConfusionMatrix<T>(List<T> uniqueTargets, T[] targets, T[] predictions)
        {
            var index = 0;
            var targetIndices = uniqueTargets.ToDictionary(t => t, t => index++);

            var targetPredictions = targets.Zip(predictions, (t, p) => new { Target = t, Prediction = p });
            var confusionMatrix = new int[uniqueTargets.Count][].Select(s => new int[uniqueTargets.Count]).ToArray();

            foreach (var targetPrediction in targetPredictions)
            {
                ++confusionMatrix[targetIndices[targetPrediction.Target]][targetIndices[targetPrediction.Prediction]];
            }

            return confusionMatrix;
        }
        
        /// <summary>
        /// Creates an error matrix based on the provided confusion matrix
        /// </summary>
        /// <param name="uniqueTargets"></param>
        /// <param name="confusionMatrix"></param>
        /// <returns></returns>
        public static double[][] ErrorMatrix<T>(List<T> uniqueTargets, int[][] confusionMatrix)
        {
            var errorMatrix = new double[uniqueTargets.Count][].Select(s => new double[uniqueTargets.Count]).ToArray();
            for (int row = 0; row < uniqueTargets.Count; ++row)
            {
                var rowSum = confusionMatrix[row].Sum();
                for (int col = 0; col < uniqueTargets.Count; col++)
                {
                    double fraction = rowSum > 0 ? ((double)confusionMatrix[row][col]) / rowSum : 0;
                    errorMatrix[row][col] = fraction;
                }
            }

            return errorMatrix;
        }
    }
}
