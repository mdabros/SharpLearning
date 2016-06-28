using MathNet.Numerics.LinearAlgebra;
using System;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Rectified linear activation for neural net.
    /// </summary>
    public sealed class ReluActivation : IActivation
    {
        /// <summary>
        /// Rectified linear activation for neural net.
        /// </summary>
        /// <param name="x"></param>
        public void Activation(Matrix<float> x)
        {
            //for (int i = 0; i < x.RowCount; i++)
            Parallel.For(0, x.RowCount, i =>
            {
                for (int j = 0; j < x.ColumnCount; j++)
                {
                    x[i, j] = Relu(x[i, j]);
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

        float Relu(float input)
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
