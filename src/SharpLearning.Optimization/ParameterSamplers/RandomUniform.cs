using System;

namespace SharpLearning.Optimization.ParameterSamplers
{
    /// <summary>
    /// Sample values random uniformly between min and max.    
    /// </summary>
    public class RandomUniform : IParameterSampler
    {
        readonly Random m_random;

        /// <summary>
        /// Sample values random uniformly between min and max. 
        /// </summary>
        /// <param name="seed"></param>
        public RandomUniform(int seed = 343)
        {
            m_random = new Random(seed);
        }

        /// <summary>
        /// Sample values random uniformly between min and max.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public double Sample(double min, double max)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }
            return m_random.NextDouble() * (max - min) + min;
        }
    }
}
