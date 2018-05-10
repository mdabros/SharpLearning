namespace SharpLearning.XGBoost
{
    /// <summary>
    /// 
    /// </summary>
    public enum ClassificationObjective
    {
        /// <summary>
        /// logistic regression for binary classification, output probability.
        /// </summary>
        BinaryLogistic,

        /// <summary>
        /// logistic regression for binary classification, output score before logistic transformation.
        /// </summary>
        BinaryLogisticRaw,

        /// <summary>
        /// set XGBoost to do multiclass classification using the softmax objective.
        /// </summary>
        Softmax,
    }
}
