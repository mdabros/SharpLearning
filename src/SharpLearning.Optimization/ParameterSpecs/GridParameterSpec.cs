using System;
using System.Linq;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// GridParameterSpec, usable when a fixed set of parameters,
    /// needs to be searched.
    /// </summary>
    public sealed class GridParameterSpec : IParameterSpec
    {
        readonly double[] m_parameters;
        readonly int m_minIndex;
        readonly int m_maxIndex;

        const ParameterType m_parameterType = ParameterType.Discrete;

        /// <summary>
        /// GridParameterSpec, usable when a fixed set of parameters,
        /// needs to be searched.
        /// </summary>
        /// <param name="parameters"></param>
        public GridParameterSpec(params double[] parameters)
        {
            m_parameters = parameters ?? throw new ArgumentNullException(nameof(parameters));
            m_minIndex = 0;
            m_maxIndex = parameters.Length - 1;
        }

        /// <summary>
        /// Minimum of the parameter range.
        /// </summary>
        public double Min => m_parameters.Min();

        /// <summary>
        /// Maximum of the parameter range.
        /// </summary>
        public double Max => m_parameters.Max();

        /// <summary>
        /// Samples a value defined for the parameter.
        /// </summary>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public double SampleValue(IParameterSampler sampler)
        {
            // sample random parameter index.
            var index = (int)sampler.Sample(m_minIndex, m_maxIndex, m_parameterType);
            // return the values of the index.
            return m_parameters[index];
        }

        /// <summary>
        /// Returns all values defined for the parameter.
        /// Primarily used for grid search.
        /// </summary>
        /// <returns></returns>
        public double[] GetAllValues()
        {
            return m_parameters;
        }
    }
}
