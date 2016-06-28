using MathNet.Numerics.LinearAlgebra;
using System;
using System.Threading.Tasks;

namespace SharpLearning.Neural.Activations
{
    /// <summary>
    /// Softmax activation for neural net.
    /// </summary>
    public sealed class SoftMaxActivation : IActivation
    {
        /// <summary>
        /// Softmax activation for neural net.
        /// </summary>
        /// <param name="x"></param>
        public void Activation(Matrix<float> x)
        {
            Parallel.For(0, x.RowCount, i =>
            {
                var rowSum = 0.0f;
                var max = double.MinValue;

                for (int j = 0; j < x.ColumnCount; ++j)
                {
                    var value = x[i, j];
                    if (value > max)
                    {
                        max = value;
                    }
                }

                for (int j = 0; j < x.ColumnCount; ++j)
                {
                    var value = (float)Math.Exp(x[i, j] - max);
                    rowSum += value;
                    x[i, j] = value;
                }

                for (int j = 0; j < x.ColumnCount; ++j)
                {
                    x[i, j] = x[i, j] / rowSum;
                }
            });
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

        /// <summary>
        /// Derivative is not supported by IdentityActivation
        /// </summary>
        /// <param name="x"></param>
        /// <param name="output"></param>
        public void Derivative(Matrix<float> x, Matrix<float> output)
        {
            throw new NotImplementedException();
        }
    }
}
