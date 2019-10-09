using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.Neural.Activations
{
    [Serializable]
    public sealed class SigmoidActivation : IActivation
    {
        /// <summary>
        /// Sigmoid activation for neural net and linear models.
        /// </summary>
        /// <param name="x"></param>
        public void Activation(float[] x)
        {
            for (int j = 0; j < x.Length; j++)
            {
                x[j] = Sigmoid(x[j]);
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

        float Sigmoid(float input)
        {
            return Convert.ToSingle(1 / (1 + Math.Pow(Math.E, -input)));
        }

        //this input should be already activated input = sigmmoid(x)
        float Derivative(float input)
        {
            var de = input * (1- input);
            
            return de == 0 ? 1 : de; //this avoid the 0 multiplication when dx is 0.
        }
    }
}
