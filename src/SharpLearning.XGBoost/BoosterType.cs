namespace SharpLearning.XGBoost
{
    /// <summary>
    /// XGBoost booster types.
    /// </summary>
    public enum BoosterType
    {
        /// <summary>
        /// Gradient boosted decision trees.
        /// </summary>
        GBTree,

        /// <summary>
        /// Gradient boosted linear models.
        /// </summary>
        GBLinear,

        /// <summary>
        /// DART: Dropouts meet Multiple Additive Regression Trees.
        /// http://xgboost.readthedocs.io/en/latest/tutorials/dart.html
        /// </summary>
        DART
    }
}
