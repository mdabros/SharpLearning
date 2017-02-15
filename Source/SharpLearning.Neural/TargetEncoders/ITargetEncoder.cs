using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.TargetEncoders
{
    /// <summary>
    /// Interface for target encoding for neural net learners.
    /// </summary>
    public interface ITargetEncoder
    {
        /// <summary>
        /// Encodes the target vector to a format accepted by a neural net learner.
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        Matrix<float> Encode(double[] targets);
    }
}
