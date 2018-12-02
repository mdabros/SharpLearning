using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Interface for parameter bounds.
    /// </summary>
    public interface IParameter
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
        /// samples a value.
        /// </summary>
        /// <param name="sampler"></param>
        /// <returns></returns>
        double SampleValue(IParameterSampler sampler);

        /// <summary>
        /// Returns all available values.
        /// Primarily used for grid search.
        /// </summary>
        /// <returns></returns>
        double[] GetAllValues();
    }
}