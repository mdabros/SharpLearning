using MathNet.Numerics.LinearAlgebra;
using System;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Tanh activation for neural net.
    /// </summary>
    public sealed class TanhActivation : IActivation
    {
        /// <summary>
        /// Tanh activation for neural net.
        /// </summary>
        /// <param name="x"></param>
        public void Activation(Matrix<float> x)
        {
            //for (int i = 0; i < x.RowCount; i++)
            Parallel.For(0, x.RowCount, i =>
            {
                for (int j = 0; j < x.ColumnCount; j++)
                {
                    x[i, j] = Tanh(x[i, j]);
                }
            });
        }

        /// <summary>
        /// Calculates the derivative and stores the result in output
        /// </summary>
        /// <param name="x"></param>
        /// <param name="output"></param>
        public void Derivative(Matrix<float> x, Matrix<float> output)
        {
            //for (int i = 0; i < x.RowCount; i++)
            Parallel.For(0, x.RowCount, i =>
            {
                for (int j = 0; j < x.ColumnCount; j++)
                {
                    output[i, j] = Derivative(x[i, j]);
                }
            });
        }

        float Tanh(float input)
        {
            if (input < -20.0f) return -1.0f; // approximation is correct to 30 decimals
            else if (input > 20.0f) return 1.0f;
            else return (float)Math.Tanh(input);
        }

        float Derivative(float input)
        {
            return (1f - (input * input));
        }

        /// <summary>
        /// Based on the fan-in and fan-out of the layer. 
        /// Determines the initialization bounds for the weights in the layer.
        /// </summary>
        /// <param name="fanIn"></param>
        /// <param name="fanOut"></param>
        /// <returns></returns>
        public float InitializationBound(int fanIn, int fanOut)
        {
            return (float)Math.Sqrt(6f / (float)(fanIn + fanOut));
        }
    }
}
