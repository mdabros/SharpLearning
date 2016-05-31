namespace SharpLearning.AdaBoost.Learners
{
    /// <summary>
    /// Loss type for adaboost regressor
    /// </summary>
    public enum AdaBoostRegressionLoss
    {
        /// <summary>
        /// Linear loss
        /// </summary>
        Linear,

        /// <summary>
        /// Squared loss
        /// </summary>
        Squared,

        /// <summary>
        /// Exponential loss
        /// </summary>
        Exponential
    }
}
