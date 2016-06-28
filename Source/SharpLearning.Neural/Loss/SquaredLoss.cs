using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.Loss
{
    /// <summary>
    /// Square loss for for neuralnet learner.
    /// The square loss function is the standard method of fitting regression models.
    /// The square loss is however sensitive to outliers since it weighs larger errors more heavily than small ones.
    /// </summary>
    public sealed class SquaredLoss : ILoss
    {
        /// <summary>
        /// return the square loss
        /// The square loss function is the standard method of fitting regression models.
        /// The square loss is however sensitive to outliers since it weighs larger errors more heavily than small ones.
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public float Loss(Matrix<float> targets, Matrix<float> predictions)
        {
            var sum = 0.0f;
            for (int i = 0; i < targets.RowCount; i++)
            {
                for (int j = 0; j < targets.ColumnCount; j++)
                {
                    var error = (targets[i, j] - predictions[i, j]);
                    sum += error * error;
                }
            }

            // 0.5 * (sum / (float)(targets.RowCount * targets.ColumnCount));
            return (sum / (float)(targets.RowCount * targets.ColumnCount));
        }
    }
}
