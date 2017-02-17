using MathNet.Numerics.LinearAlgebra;

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
    }
}
