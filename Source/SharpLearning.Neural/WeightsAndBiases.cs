using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Neural
{
    /// <summary>
    /// 
    /// </summary>
    public class WeightsAndBiases
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly Matrix<float> Weights;

        /// <summary>
        /// 
        /// </summary>
        public readonly Vector<float> Bias;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="bias"></param>
        public WeightsAndBiases(Matrix<float> weights, Vector<float> bias)
        {
            Weights = weights;
            Bias = bias;
        }
    }
}
