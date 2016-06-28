using MathNet.Numerics.LinearAlgebra;
using System;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Logistic activation function for neural net.
    /// </summary>
    public sealed class LogisticActiviation : IActivation
    {
        /// <summary>
        /// Logistic activation function for neural net.
        /// </summary>
        /// <param name="x"></param>
        public void Activation(Matrix<float> x)
        {
            Parallel.For(0, x.RowCount, i =>
            {
                for (int j = 0; j < x.ColumnCount; j++)
                {
                    x[i, j] = LogisticSigmoid(x[i, j]);
                }
            });
        }


        float LogisticSigmoid(float value)
        {
            return 1f / (1f + (float)Math.Exp(-value));
        }

        float Derivative(float value)
        {
            return value * (1 - value);
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

        /// <summary>
        /// Based on the fan-in and fan-out of the layer. 
        /// Determines the initialization bounds for the weights in the layer.
        /// </summary>
        /// <param name="fanIn"></param>
        /// <param name="fanOut"></param>
        /// <returns></returns>
        public float InitializationBound(int fanIn, int fanOut)
        {
            return (float)Math.Sqrt(2f / (float)(fanIn + fanOut));
        }
    }
}
