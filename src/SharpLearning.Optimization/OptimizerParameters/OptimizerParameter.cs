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
        public readonly ParameterSampler Sampler;

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
            Sampler = sampler;
        }

        /// <summary>
        /// Contains the bounds and sampling type for an optimizer parameter.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="samplerType">Selects between predefined sample types for controlling how to sample between the specified min and max bounds.
        /// Default is Linear.</param>
        public OptimizerParameter(double min, double max, ParameterSamplerType samplerType = ParameterSamplerType.Linear)
            : this(min, max, RandomUniform.Create(samplerType))
        {
        }
    }
}
