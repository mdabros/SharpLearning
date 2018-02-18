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

        /// <summary>
        /// Contains the bounds and transform type for an optimizer.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="transform">Selects between predefined transform types for controlling how to scale values sampled between min and max bounds.
        /// Default is Linear.</param>
        public ParameterBounds(double min, double max, Transform transform = Transform.Linear)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }

            Min = min;
            Max = max;
            m_transform = TransformFactory.Create(transform);
        }

        /// <summary>
        /// Contains the bounds and transform type for an optimizer.
        /// </summary>
        /// <param name="min">minimum bound.</param>
        /// <param name="max">maximum bound.</param>
        /// <param name="transform">Transform for controlling the scale of the parameter sampled between min and max bounds.</param>
        public ParameterBounds(double min, double max, ITransform transform)
        {
            if (min >= max) { throw new ArgumentException($"min: {min} is larger than or equal to max: {max}"); }
            if (transform == null) throw new ArgumentNullException(nameof(transform));

            Min = min;
            Max = max;
            m_transform = transform;
        }


        /// <summary>
        /// Samples a new point within the specified parameter bounds.
        /// </summary>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public double NextValue(IParameterSampler sampler)
        {
            return m_transform.Transform(Min, Max, sampler);
        }
    }
}
