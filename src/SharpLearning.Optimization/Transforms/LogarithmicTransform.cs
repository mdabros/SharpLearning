﻿using System;
using SharpLearning.Optimization.ParameterSamplers;

namespace SharpLearning.Optimization.Transforms
{
    /// <summary>
    /// Logarithmic scale. For larger ranges like min: 0.0001 and max: 1.0.
    /// </summary>
    public class LogarithmicTransform : ITransform
    {
        /// <summary>
        /// Logarithmic scale. For larger ranges like min: 0.0001 and max: 1.0.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="sampler"></param>
        /// <returns></returns>
        public double Transform(double min, double max, IParameterSampler sampler)
        {
            if (min <= 0 || max <= 0) { throw new ArgumentException($"logarithmic scale requires min: {min} and max: {max} to be larger than zero"); }
            var a = Math.Log10(min);
            var b = Math.Log10(max);

            var r = sampler.Sample(a, b);
            return Math.Pow(10, r);
        }
    }
}
