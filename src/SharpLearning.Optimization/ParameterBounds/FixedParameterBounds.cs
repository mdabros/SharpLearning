using System;
using System.Linq;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Fixed parameter bounds, usable when a fixed set of parameters,
    /// needs to be searched.
    /// </summary>
    public sealed class FixedParameterBounds : IParameterBounds
    {
        readonly double[] m_parameters;

        /// <summary>
        /// Fixed parameter bounds, usable when a fixed set of parameters,
        /// needs to be searched.
        /// </summary>
        /// <param name="parameters"></param>
        public FixedParameterBounds(double[] parameters)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
        }

        /// <summary>
        /// Maximum of the parameter range.
        /// </summary>
        public double Max => m_parameters.Max();

        /// <summary>
        /// Minimum of the parameter range.
        /// </summary>
        public double Min => m_parameters.Min();

        /// <summary>
        /// Gets the next value from the parameter range.
        /// </summary>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public double NextValue(IParameterSampler sampler)
        {
            throw new NotImplementedException();
        }
    }
}
