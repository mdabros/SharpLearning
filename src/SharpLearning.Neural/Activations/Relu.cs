using System;

namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Rectified linear activation for neural net.
    /// </summary>
    [Serializable]
    public sealed class Relu : INonLinearity
    {
        /// <summary>
        /// Rectified linear activation for neural net.
        /// </summary>
        /// <param name="x"></param>
        public void Activation(float[] x)
        {
            for (int j = 0; j < x.Length; j++)
            {
                x[j] = ReluFunc(x[j]);
            }
        }

        /// <summary>
        /// Calculates the derivative and stores the result in output
        /// </summary>
        /// <param name="x"></param>
        /// <param name="output"></param>
        public void Derivative(float[] x, float[] output)
        {
            for (int j = 0; j < x.Length; j++)
            {
                output[j] = Derivative(x[j]);
            }
        }

        float ReluFunc(float input)
        {
            return Math.Max(0, input);
        }

        float Derivative(float input)
        {
            if (input > 0.0)
                return 1.0f;
            else
                return 0.0f;
        }
    }
}
