using MathNet.Numerics.LinearAlgebra;

namespace SharpLearning.Neural
{
    /// <summary>
    /// 
    /// </summary>
    public class ParametersAndGradients
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly WeightsAndBiases Parameters;

        /// <summary>
        /// 
        /// </summary>
        public readonly WeightsAndBiases Gradients;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="gradients"></param>
        public ParametersAndGradients(WeightsAndBiases parameters, WeightsAndBiases gradients)
        {
            Parameters = parameters;
            Gradients = gradients;
        }
    }
}
