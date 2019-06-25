using System;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.Optimization.Transforms;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// MinMaxParameterSpec, used for sampling values in the range [min;max].
    /// </summary>
    public class MinMaxParameterSpec : IParameterSpec
    {
        readonly ITransform m_transform;

        readonly ParameterType m_parameterType;

        /// <summary>
        /// MinMaxParameterSpec, used for sampling values in the range [min;max].
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="transform">Selects between predefined transform types for controlling how to scale values sampled between min and max bounds.
        /// Default is Linear.</param>
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continuous values.
        /// Default is Continuous.</param>
        public MinMaxParameterSpec(double min, double max, 
            Transform transform = Transform.Linear, ParameterType parameterType  = ParameterType.Continuous)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }

            Min = min;
            Max = max;
            m_transform = TransformFactory.Create(transform);
            m_parameterType = parameterType;
        }

        /// <summary>
        /// MinMaxParameterSpec, used for sampling values in the range [min;max].
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="transform">Transform for controlling the scale of the parameter sampled between min and max bounds.</param>
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continous values.</param>
        public MinMaxParameterSpec(double min, double max, 
            ITransform transform, ParameterType parameterType)
        {
            if (min >= max)
            {
                throw new ArgumentException($"min: {min} is larger than or equal to max: {max}");
            }

            m_transform = transform ?? throw new ArgumentNullException(nameof(transform));

            Min = min;
            Max = max;
            m_parameterType = parameterType;
        }

        /// <summary>
        /// Minimum bound.
        /// </summary>
        public double Min { get; }

        /// <summary>
        /// Maximum bound.
        /// </summary>
        public double Max { get; }

        /// <summary>
        /// Samples a new value within the specified parameter bounds.
        /// </summary>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public double SampleValue(IParameterSampler sampler)
        {
            return m_transform.Transform(Min, Max, m_parameterType, sampler);
        }

        /// <summary>
        /// Not available for MinMaxParameterSpec.
        /// </summary>
        /// <returns></returns>
        public double[] GetAllValues()
        {
            throw new NotImplementedException($"Get all values is not available for {nameof(MinMaxParameterSpec)}");
        }
    }
}
