using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Interface for parameter bounds.
    /// </summary>
    public interface IParameterBounds
    {
        /// <summary>
        /// Minimum bound.
        /// </summary>
        double Max { get; }

        /// <summary>
        /// Maximum bound.
        /// </summary>
        double Min { get; }

        /// <summary>
        /// Get next value.
        /// </summary>
        /// <param name="sampler"></param>
        /// <returns></returns>
        double NextValue(IParameterSampler sampler);
    }
}