using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Containers.Tensors;

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
            return Loss(targets.Data(), predictions.Data());
        }

        /// <summary>
        /// return the square loss
        /// The square loss function is the standard method of fitting regression models.
        /// The square loss is however sensitive to outliers since it weighs larger errors more heavily than small ones.
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public float Loss(Tensor<float> targets, Tensor<float> predictions)
        {
            return Loss(targets.Data, predictions.Data);
        }

        /// <summary>
        /// return the square loss
        /// The square loss function is the standard method of fitting regression models.
        /// The square loss is however sensitive to outliers since it weighs larger errors more heavily than small ones.
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        public double Loss(Tensor<double> targets, Tensor<double> predictions)
        {
            return Loss(targets.Data, predictions.Data);
        }

        float Loss(float[] targetsData, float[] predictionsData)
        {
            var sum = 0.0f;

            for (int i = 0; i < targetsData.Length; i++)
            {
                var error = (targetsData[i] - predictionsData[i]);
                sum += error * error;
            }

            return 0.5f * (sum / (float)(targetsData.Length));
        }

        double Loss(double[] targetsData, double[] predictionsData)
        {
            var sum = 0.0;

            for (int i = 0; i < targetsData.Length; i++)
            {
                var error = (targetsData[i] - predictionsData[i]);
                sum += error * error;
            }

            return 0.5f * (sum / (double)(targetsData.Length));
        }
    }
}
