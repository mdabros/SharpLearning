using System;

namespace SharpLearning.Optimization.ParameterSamplers
{
    public sealed class RandomUniformIntergers : IParameterSampler
    {
        readonly Random m_random;

        /// <summary>
        /// Sample values random uniformly between min and max. 
        /// </summary>
        /// <param name="seed"></param>
        public RandomUniformIntergers(int seed = 343)
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
            var minInt = (int)min;
            var maxInt = (int)(max + 1); // Add one to get inclusive.

            return m_random.Next(minInt, maxInt);
        }
    }
}
