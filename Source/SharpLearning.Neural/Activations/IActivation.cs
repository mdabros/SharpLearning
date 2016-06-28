using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Neural net activiation interface
    /// </summary>
    public interface IActivation
    {
        /// <summary>
        /// Calcualtes the activation and stores the result in x
        /// </summary>
        /// <param name="x"></param>
        void Activation(Matrix<float> x);

        /// <summary>
        /// Calculates the derivative and stores the result in output
        /// </summary>
        /// <param name="x"></param>
        /// <param name="output"></param>
        void Derivative(Matrix<float> x, Matrix<float> output);

        /// <summary>
        /// Based on the fan-in and fan-out of the layer. 
        /// Determines the initialization bounds for the weights in the layer.
        /// </summary>
        /// <param name="fanIn"></param>
        /// <param name="fanOut"></param>
        /// <returns></returns>
        float InitializationBound(int fanIn, int fanOut);
    }
}
