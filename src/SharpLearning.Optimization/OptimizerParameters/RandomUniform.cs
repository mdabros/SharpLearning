using System;

namespace SharpLearning.Optimization.OptimizerParameters
{
    /// <summary>
    /// Class for random uniform sampling.
    /// </summary>
    public static class RandomUniform
    {
        /// <summary>
        /// Return a ParameterSampler delegate based on the parameterSamplerType
        /// </summary>
        /// <param name="parameterSamplerType"></param>
        /// <returns></returns>
        public static ParameterSampler Create(ParameterSamplerType parameterSamplerType)
        {
            switch (parameterSamplerType)
            {
                case ParameterSamplerType.Linear:
                    return (min, max, random) => Linear(min, max, random);
                case ParameterSamplerType.Logarithmic:
                    return (min, max, random) => Logarithmic(min, max, random);
                default:
                    throw new ArgumentException("Unsupported ParameterSamplerType: " + parameterSamplerType);
            }
        }

        /// <summary>
        /// Sample values random uniformly between min and max on a linear scale.
        /// For smaller ranges like min: 64 and max: 256
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static double Linear(double min, double max, Random random)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }

            return random.NextDouble() * (max - min) + min;
        }

        /// <summary>
        /// Sample values random uniformly between min and max on a logarithmic scale. 
        /// For larger ranges like min: 0.0001 and max: 1.0.
        /// This requires min and max to be larger than zero.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public static double Logarithmic(double min, double max, Random random)
        {
            if (min <= 0 || max <= 0) { throw new ArgumentException($"logarithmic scale requires min: {min} and max: {max} to be larger than zero"); }
            var a = Math.Log10(min);
            var b = Math.Log10(max);

            var r = Linear(a, b, random);
            return Math.Pow(10, r);
        }
    }
}
