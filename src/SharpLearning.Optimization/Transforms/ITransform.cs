using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Transforms
{
    /// <summary>
    /// Interface for transforms. 
    /// </summary>
    public interface ITransform
    {
        /// <summary>
        /// Adds a transform to the values used in the sampler.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="sampler"></param>
        /// <returns></returns>
        double Transform(double min, double max, IParameterSampler sampler);
    }
}
