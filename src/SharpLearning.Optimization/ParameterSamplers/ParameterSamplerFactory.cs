using System;

namespace SharpLearning.Optimization.ParameterSamplers
{
    /// <summary>
    /// Factory for ParameterSampler 
    /// </summary>
    public static class ParameterSamplerFactory
    {
        /// <summary>
        /// Return a ParameterSampler based on the predefined selection.
        /// </summary>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public static IParameterSampler Create(ParameterSamplerType sampler)
        {
            switch (sampler)
            {
                case ParameterSamplerType.RandomUniformLinear:
                    return new RandomUniformLinear();
                case ParameterSamplerType.RandomUniformLogarithmic:
                    return new RandomUniformLogarithmic();
                default:
                    throw new ArgumentException("Unsupported ParameterSampler: " + sampler);
            }
        }
    }
}
