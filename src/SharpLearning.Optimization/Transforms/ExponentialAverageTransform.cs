using System;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Transforms
{
    /// <summary>
    /// ExponentialAverage scale. For ranges close to one, like min: 0.9 and max: 0.999.
    /// Note that the min and max must be smaller than 1 for this transform.
    /// </summary>
    public class ExponentialAverageTransform : ITransform
    {
        /// <summary>
        /// ExponentialAverage scale. For ranges close to one, like min: 0.9 and max: 0.999.
        /// Note that the min and max values must be smaller than 1 for this transform.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="sampler"></param>
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continous values.</param>
        /// <returns></returns>
        public double Transform(double min, double max, ParameterType parameterType, IParameterSampler sampler)
        {
            if (min >= 1 || max >= 1)
            {
                throw new ArgumentException("ExponentialAverage scale requires min: " + 
                    $" {min} and max: {max} to be smaller than one");
            }

            var a = Math.Log10(1 - max);
            var b = Math.Log10(1 - min);

            var r = sampler.Sample(a, b, parameterType);
            return 1.0 - Math.Pow(10, r);
        }
    }
}
