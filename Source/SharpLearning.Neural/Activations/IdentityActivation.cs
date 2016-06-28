using MathNet.Numerics.LinearAlgebra;
using System;

namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Identity activation for neural net. This activation function simply does nothing
    /// </summary>
    public class IdentityActivation : IActivation
    {
        /// <summary>
        /// Identity activation for neural net. This activation function simply does nothing
        /// </summary>
        /// <param name="x"></param>
        public void Activation(Matrix<float> x)
        {
            // do nothing and just leave the input.
        }

        /// <summary>
        /// Derivative is not supported by IdentityActivation
        /// </summary>
        /// <param name="x"></param>
        /// <param name="output"></param>
        public void Derivative(Matrix<float> x, Matrix<float> output)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Initializatin bounds is not supported by IdentityActivation
        /// </summary>
        /// <param name="fanIn"></param>
        /// <param name="fanOut"></param>
        /// <returns></returns>
        public float InitializationBound(int fanIn, int fanOut)
        {
            throw new NotImplementedException();
        }
    }
}
