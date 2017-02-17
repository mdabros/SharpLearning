using MathNet.Numerics.LinearAlgebra;

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
    }
}
