namespace SharpLearning.XGBoost;

/// <summary>
/// Sampler type for DART
/// </summary>
public enum SamplerType
{
    /// <summary>
    /// Dropped trees are selected uniformly.
    /// </summary>
    Uniform,

    /// <summary>
    /// Dropped trees are selected in proportion to weight.
    /// </summary>
    Weighted,
}
