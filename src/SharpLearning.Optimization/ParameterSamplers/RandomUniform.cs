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
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continous values.</param>
        /// <returns></returns>
        public double Sample(double min, double max, ParameterType parameterType)
        {
            if (min >= max)
            {
                throw new ArgumentException($"min: {min} is larger than or equal to max: {max}");
            }

            switch (parameterType)
            {
                case ParameterType.Discrete:
                    return SampleInteger((int)min, (int)max);
                case ParameterType.Continuous:
                    return SampleContinous(min, max);
                default:
                    throw new ArgumentException("Unknown parameter type: " + parameterType);
            }
        }

        double SampleContinous(double min, double max)
        {
            return m_random.NextDouble() * (max - min) + min;
        }

        int SampleInteger(int min, int max)
        {
            var maxInclusive = max + 1; // Add one to get inclusive.
            return m_random.Next(min, maxInclusive);
        }
    }
}
