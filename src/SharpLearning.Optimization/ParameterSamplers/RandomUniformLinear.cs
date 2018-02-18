using System;

namespace SharpLearning.Optimization.ParameterSamplers
{
    /// <summary>
    /// Sample values random uniformly between min and max on a linear scale.
    /// For smaller ranges like min: 64 and max: 256
    /// </summary>
    public class RandomUniformLinear : IParameterSampler
    {
        /// <summary>
        /// Sample values random uniformly between min and max on a linear scale.
        /// For smaller ranges like min: 64 and max: 256
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public double Sample(double min, double max, Random random)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }
            return random.NextDouble() * (max - min) + min;
        }
    }
}
