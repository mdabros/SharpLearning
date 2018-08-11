using System;
using SharpLearning.Optimization.ParameterSamplers;
using SharpLearning.Optimization.Transforms;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Contains the bounds and sampling type for an optimizer parameter.
    /// </summary>
    public class ParameterBounds
    {
        /// <summary>
        /// Minimum bound.
        /// </summary>
        public readonly double Min;

        /// <summary>
        /// Maximum bound.
        /// </summary>
        public readonly double Max;

        readonly ITransform m_transform;

        readonly ParameterType m_parameterType;

        /// <summary>
        /// Contains the bounds and transform type for an optimizer.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="transform">Selects between predefined transform types for controlling how to scale values sampled between min and max bounds.
        /// Default is Linear.</param>
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continous values.
        /// Default is Continous.</param>
        public ParameterBounds(double min, double max, 
            Transform transform = Transform.Linear, ParameterType parameterType  = ParameterType.Continuous)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }

            Min = min;
            Max = max;
            m_transform = TransformFactory.Create(transform);
            m_parameterType = parameterType;
        }

        /// <summary>
        /// Contains the bounds and transform type for an optimizer.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="transform">Transform for controlling the scale of the parameter sampled between min and max bounds.</param>
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continous values.</param>
        public ParameterBounds(double min, double max, 
            ITransform transform, ParameterType parameterType)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }
            if (transform == null) throw new ArgumentNullException(nameof(transform));

            Min = min;
            Max = max;
            m_transform = transform;
            m_parameterType = parameterType;
        }


        /// <summary>
        /// Samples a new point within the specified parameter bounds.
        /// </summary>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public double NextValue(IParameterSampler sampler)
        {
            return m_transform.Transform(Min, Max, m_parameterType, sampler);
        }
    }
}
