using System;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Transforms
{
    /// <summary>
    /// Transform to Log10 scale. For ranges with a large difference in numerical scale, like min: 0.0001 and max: 1.0.
    /// </summary>
    public class Log10Transform : ITransform
    {
        /// <summary>
        /// Transform to Log10 scale. For ranges with a large difference in numerical scale, like min: 0.0001 and max: 1.0.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="sampler"></param>
        /// <param name="parameterType">Selects the type of parameter. Should the parameter be sampled as discrete values, or as continous values.</param>
        /// <returns></returns>
        public double Transform(double min, double max, ParameterType parameterType, IParameterSampler sampler)
        {
            if (min <= 0 || max <= 0)
            {
                throw new ArgumentException("logarithmic scale requires min: " + 
                    $"{min} and max: {max} to be larger than zero");
            }
            var a = Math.Log10(min);
            var b = Math.Log10(max);

            var r = sampler.Sample(a, b, parameterType);
            return Math.Pow(10, r);
        }
    }
}
