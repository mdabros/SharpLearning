using System;

namespace SharpLearning.Optimization.ParameterSamplers
{
    /// <summary>
    /// Sample values random uniformly between min and max on a logarithmic scale. 
    /// For larger ranges like min: 0.0001 and max: 1.0.
    /// This requires min and max to be larger than zero.
    /// </summary>
    public class RandomUniformLogarithmic : IParameterSampler
    {
        readonly RandomUniformLinear m_linear = new RandomUniformLinear();

        /// <summary>
        /// Sample values random uniformly between min and max on a logarithmic scale. 
        /// For larger ranges like min: 0.0001 and max: 1.0.
        /// This requires min and max to be larger than zero.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="random"></param>
        /// <returns></returns>
        public double Sample(double min, double max, Random random)
        {
            if (min <= 0 || max <= 0) { throw new ArgumentException($"logarithmic scale requires min: {min} and max: {max} to be larger than zero"); }
            var a = Math.Log10(min);
            var b = Math.Log10(max);

            var r = m_linear.Sample(a, b, random);
            return Math.Pow(10, r);
        }
    }
}
