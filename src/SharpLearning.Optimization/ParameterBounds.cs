using System;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Contains the bounds and sampling type for an optimizer parameter.
    /// </summary>
    public class ParameterBounds
    {
        public readonly double Min;
        public readonly double Max;
        readonly IParameterSampler m_sampler;

        /// <summary>
        /// Contains the bounds and sampling type for an optimizer parameter.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="samplerType">Selects between predefined sampler types for controlling how to sample between the specified min and max bounds.
        /// Default is Linear.</param>
        public ParameterBounds(double min, double max, ParameterSamplerType samplerType = ParameterSamplerType.RandomUniformLinear)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }

            Min = min;
            Max = max;
            m_sampler = ParameterSamplerFactory.Create(samplerType);
        }

        /// <summary>
        /// Contains the bounds and sampling type for an optimizer parameter.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="samplerScale">Parameter sampler for controlling how to sample between the specified min and max bounds.</param>
        public ParameterBounds(double min, double max, IParameterSampler sampler)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }
            if (sampler == null) throw new ArgumentNullException(nameof(sampler));

            Min = min;
            Max = max;
            m_sampler = sampler;
        }

        /// <summary>
        /// Samples a new point within the specified parameter bounds.
        /// </summary>
        /// <param name="random"></param>
        /// <returns></returns>
        public double Sample(Random random)
        {
            return m_sampler.Sample(Min, Max, random);
        }
    }
}
