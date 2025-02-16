using System;

namespace SharpLearning.Optimization;

/// <summary>
/// Optimization result
/// </summary>
[Serializable]
public sealed class OptimizerResult
{
    public OptimizerResult(double[] parameterSet, double error)
    {
        ParameterSet = parameterSet ?? throw new ArgumentNullException(nameof(parameterSet));
        Error = error;
    }

    /// <summary>
    /// Resulting error using the parameter set.
    /// </summary>
    public double Error { get; }

    /// <summary>
    /// The parameter set.
    /// </summary>
    public double[] ParameterSet { get; }
}
