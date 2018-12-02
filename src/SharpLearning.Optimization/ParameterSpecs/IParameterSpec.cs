using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization
{
    /// <summary>
    /// Interface for parameter specs.
    /// </summary>
    public interface IParameterSpec
    {
        /// <summary>
        /// Minimum bound.
        /// </summary>
        double Min { get; }

        /// <summary>
        /// Maximum bound.
        /// </summary>
        double Max { get; }

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