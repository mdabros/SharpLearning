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
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continous values.</param>
        /// <returns></returns>
        double Transform(double min, double max, ParameterType parameterType, IParameterSampler sampler);
    }
}
