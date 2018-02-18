using System;

namespace SharpLearning.Optimization.OptimizerParameters
{
    /// <summary>
    /// Parameter sampler delegate. Defines the contract for a parameter sampler.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="random"></param>
    /// <returns></returns>
    public delegate double ParameterSampler(double min, double max, Random random);
}
