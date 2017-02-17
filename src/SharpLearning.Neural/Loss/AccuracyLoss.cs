using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.Loss
{
    /// <summary>
    /// Loss function for classification accuracy.
    /// https://en.wikipedia.org/wiki/Accuracy_and_precision
    /// </summary>
    public sealed class AccuracyLoss : ILoss
    {
        /// <summary>
        /// Loss function for classification accuracy.
        /// https://en.wikipedia.org/wiki/Accuracy_and_precision
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public float Loss(Matrix<float> targets, Matrix<float> predictions)
        {
            var correctCount = 0;

            for (int row = 0; row < targets.RowCount; row++)
            {
                var max = 0.0;
                var maxIndex = 0;
                for (int col = 0; col < targets.ColumnCount; col++)
                {
                    var predictionValue = predictions.At(row, col);
                    if(predictionValue > max)
                    {
                        max = predictionValue;
                        maxIndex = col;
                    }
                }

                if(targets.At(row, maxIndex) == 1.0)
                {
                    correctCount++;
                }
            }

            // returns the error - not the accuracy.
            return 1.0f - ((float)correctCount / (float)targets.RowCount);
        }
    }
}
