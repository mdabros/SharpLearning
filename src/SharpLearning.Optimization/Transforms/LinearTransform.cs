using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Transforms
{
    /// <summary>
    /// Linear scale. For smaller ranges like min: 64 and max: 256.
    /// </summary>
    public class LinearTransform : ITransform
    {
        /// <summary>
        /// Linear scale. For smaller ranges like min: 64 and max: 256.
        /// Returns the samplers value directly.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public double Transform(double min, double max, IParameterSampler sampler)
        {
            return sampler.Sample(min, max);
        }
    }
}
