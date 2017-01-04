using System;

namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Default activation bounds
    /// </summary>
    public static class ActivationInitializationBounds
    {
        /// <summary>
        /// Returns the default activation bounds based on the activation type, fan-in and fan-out.
        /// </summary>
        /// <param name="activation"></param>
        /// <param name="fanIn"></param>
        /// <param name="fanOut"></param>
        /// <returns></returns>
        public static float InitializationBound(Activation activation, int fanIn, int fanOut)
        {
            switch (activation)
            {
                case Activation.Undefined:
                    return (float)Math.Sqrt(2.0 / (double)(fanOut)); // use standard initialization for undefined.
                case Activation.Relu:
                    return (float)Math.Sqrt(2.0 / (double)(fanOut)); // according to article on Relu activation: https://arxiv.org/pdf/1502.01852.pdf
                default:
                    throw new ArgumentException("Unsupported activation type: " + activation);
            }
        }
    }
}
