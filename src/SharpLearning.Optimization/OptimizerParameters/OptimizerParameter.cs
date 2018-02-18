using System;

namespace SharpLearning.Optimization.OptimizerParameters
{
    /// <summary>
    /// Contains the bounds and sampling type for an optimizer parameter.
    /// </summary>
    public class OptimizerParameter
    {
        public readonly double Min;
        public readonly double Max;
        readonly ParameterSampler m_sampler;

        /// <summary>
        /// Contains the bounds and sampling type for an optimizer parameter.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="sampler">ParameterSampler controls how to sample between the specified min and max bounds.</param>
        public OptimizerParameter(double min, double max, ParameterSampler sampler)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }
            if (sampler == null) throw new ArgumentNullException(nameof(sampler));

            Min = min;
            Max = max;
            m_sampler = sampler;
        }

        /// <summary>
        /// Contains the bounds and sampling type for an optimizer parameter.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="samplerScale">Selects between predefined sample scales for controlling how to sample between the specified min and max bounds.
        /// Default is Linear.</param>
        public OptimizerParameter(double min, double max, SamplerScale samplerScale = SamplerScale.Linear)
            : this(min, max, RandomUniform.Create(samplerScale))
        {
        }

        public double Sample(Random random)
        {
            return m_sampler(Min, Max, random);
        }
    }
}
