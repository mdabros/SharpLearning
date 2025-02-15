namespace SharpLearning.XGBoost;

/// <summary>
/// XGBoost tree methods.
/// </summary>
public enum TreeMethod
{
    /// <summary>
    /// Auto: Use heuristic to choose faster one.
    /// - For small to medium dataset, exact greedy will be used.
    /// - For very large-dataset, approximate algorithm will be chosen.
    /// - Because old behavior is always use exact greedy in single machine,
    /// </summary>
    Auto,

    /// <summary>
    /// Exact greedy algorithm
    /// </summary>
    Exact,

    /// <summary>
    /// Approximate greedy algorithm using sketching and histogram.
    /// </summary>
    Approx,

    /// <summary>
    /// Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
    /// </summary>
    Hist,

    /// <summary>
    /// GPU implementation of exact algorithm.
    /// </summary>
    GPUExact,

    /// <summary>
    /// GPU implementation of hist algorithm.
    /// </summary>
    GPUHist,
}
