using MathNet.Numerics.LinearAlgebra;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.Loss
{
    /// <summary>
    /// Interface for neuralnet learner
    /// </summary>
    public interface ILoss
    {
        /// <summary>
        /// Returns the loss
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        float Loss(Matrix<float> targets, Matrix<float> predictions);


        /// <summary>
        /// Returns the loss
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        float Loss(Tensor<float> targets, Tensor<float> predictions);

        /// <summary>
        /// Returns the loss
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        double Loss(Tensor<double> targets, Tensor<double> predictions);
    }
}
