using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Transforms
{
    /// <summary>
    /// Linear scale. For ranges with a small difference in numerical scale, like min: 64 and max: 256.
    /// </summary>
    public class LinearTransform : ITransform
    {
        /// <summary>
        /// Linear scale. For ranges with a small difference in numerical scale, like min: 64 and max: 256.
        /// Returns the samplers value directly.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="sampler"></param>
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continous values.</param>
        /// <returns></returns>
        public double Transform(double min, double max, ParameterType parameterType, IParameterSampler sampler)
        {
            return sampler.Sample(min, max, parameterType);
        }
    }
}
