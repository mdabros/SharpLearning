using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.Loss
{
    /// <summary>
    /// Square loss for for neuralnet learner.
    /// The square loss function is the standard method of fitting regression models.
    /// The square loss is however sensitive to outliers since it weighs larger errors more heavily than small ones.
    /// </summary>
    public sealed class SquareLoss : ILoss
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

            var targetsArray = targets.Data();
            var predictionsArray = predictions.Data();

            for (int i = 0; i < targetsArray.Length; i++)
            {
                var error = (targetsArray[i] - predictionsArray[i]);
                sum += error * error;
            }

            return 0.5f * (sum / (float)(targets.RowCount * targets.ColumnCount));
        }
    }
}
