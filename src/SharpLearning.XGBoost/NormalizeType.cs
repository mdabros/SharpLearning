namespace SharpLearning.XGBoost;

/// <summary>
/// Type of normalization algorithm for DART
/// </summary>
public enum NormalizeType
{
    /// <summary>
    /// New trees have the same weight of each of dropped trees.
    /// Weight of new trees are 1 / (k + learning_rate).
    /// Dropped trees are scaled by a factor of k / (k + learning_rate).
    /// </summary>
    Tree,

    /// <summary>
    /// New trees have the same weight of sum of dropped trees(forest).
    /// Weight of new trees are 1 / (1 + learning_rate).
    /// Dropped trees are scaled by a factor of 1 / (1 + learning_rate)
    /// </summary>
    Forest
}
