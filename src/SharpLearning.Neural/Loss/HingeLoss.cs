using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Loss
{
    /// <summary>
    /// Hinge loss, used by linear svm
    /// https://en.wikipedia.org/wiki/Hinge_loss
    /// </summary>
    public sealed class HingeLoss : ILoss
    {
        /// <summary>
        /// Hinge loss, used by linear svm
        /// https://en.wikipedia.org/wiki/Hinge_loss
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public float Loss(Matrix<float> targets, Matrix<float> predictions)
        {
            const double margin = 1.0;
            var batchSize = targets.RowCount;
            var loss = 0.0;

            for (int batchItem = 0; batchItem < batchSize; batchItem++)
            {
                var maxTarget = 0.0;
                var maxTargetIndex = 0;
                for (int col = 0; col < targets.ColumnCount; col++)
                {
                    var targetValue = targets.At(batchItem, col);
                    if (targetValue > maxTarget)
                    {
                        maxTarget = targetValue;
                        maxTargetIndex = col;
                    }
                }

                var maxTargetScore = predictions.At(batchItem, maxTargetIndex);
                for (int i = 0; i < predictions.ColumnCount; i++)
                {
                    if (i == maxTargetIndex) { continue; }

                    // The score of the target should be higher than he score of any other class, by a margin
                    var diff = -maxTargetScore + predictions.At(batchItem, i) + margin;
                    if (diff > 0)
                    {
                        loss += diff;
                    }
                }
            }

            return (float)(loss / (double)batchSize);
        }

        /// <summary>
        /// Hinge loss, used by linear svm
        /// https://en.wikipedia.org/wiki/Hinge_loss
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public float Loss(Tensor<float> targets, Tensor<float> predictions)
        {
            const double margin = 1.0;
            var rows = targets.Dimensions[0];
            var cols = targets.Dimensions[1];
            var loss = 0.0;

            var targetsData = targets.Data;
            var predictionsData = predictions.Data;

            for (int batchItem = 0; batchItem < rows; batchItem++)
            {
                var maxTarget = 0.0;
                var maxTargetIndex = 0;
                var rowOffSet = batchItem * cols;
                for (int col = 0; col < cols; col++)
                {
                    var index = rowOffSet + cols;
                    var targetValue = targetsData[index];
                    if (targetValue > maxTarget)
                    {
                        maxTarget = targetValue;
                        maxTargetIndex = col;
                    }
                }

                var maxPredictionIndex = rowOffSet + maxTargetIndex;
                var maxTargetScore = predictionsData[maxPredictionIndex];
                for (int i = 0; i < cols; i++)
                {
                    if (i == maxTargetIndex) { continue; }

                    // The score of the target should be higher than he score of any other class, by a margin
                    var index = rowOffSet + i;
                    var diff = -maxTargetScore + predictionsData[index] + margin;
                    if (diff > 0)
                    {
                        loss += diff;
                    }
                }
            }

            return (float)(loss / (double)rows);
        }

        /// <summary>
        /// Hinge loss, used by linear svm
        /// https://en.wikipedia.org/wiki/Hinge_loss
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Loss(Tensor<double> targets, Tensor<double> predictions)
        {
            const double margin = 1.0;
            var rows = targets.Dimensions[0];
            var cols = targets.Dimensions[1];
            var loss = 0.0;

            var targetsData = targets.Data;
            var predictionsData = predictions.Data;

            for (int batchItem = 0; batchItem < rows; batchItem++)
            {
                var maxTarget = 0.0;
                var maxTargetIndex = 0;
                var rowOffSet = batchItem * cols;
                for (int col = 0; col < cols; col++)
                {
                    var index = rowOffSet + cols;
                    var targetValue = targetsData[index];
                    if (targetValue > maxTarget)
                    {
                        maxTarget = targetValue;
                        maxTargetIndex = col;
                    }
                }

                var maxPredictionIndex = rowOffSet + maxTargetIndex;
                var maxTargetScore = predictionsData[maxPredictionIndex];
                for (int i = 0; i < cols; i++)
                {
                    if (i == maxTargetIndex) { continue; }

                    // The score of the target should be higher than he score of any other class, by a margin
                    var index = rowOffSet + i;
                    var diff = -maxTargetScore + predictionsData[index] + margin;
                    if (diff > 0)
                    {
                        loss += diff;
                    }
                }
            }

            return (double)(loss / (double)rows);
        }
    }
}
