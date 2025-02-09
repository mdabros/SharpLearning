namespace SharpLearning.XGBoost;

/// <summary>
/// Classification objectives.
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
    /// GPU version of binary logistic regression evaluated on the GPU,
    /// note that like the GPU histogram algorithm, 
    /// they can only be used when the entire training session uses the same dataset.
    /// </summary>
    GPUBinaryLogistic,

    /// <summary>
    /// GPU version of binary logistic regression raw evaluated on the GPU,
    /// note that like the GPU histogram algorithm, 
    /// they can only be used when the entire training session uses the same dataset.
    /// </summary>
    GPUBinaryLogisticRaw,

    /// <summary>
    /// Multiclass classification using the softmax objective.
    /// </summary>
    Softmax,

    /// <summary>
    /// same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, 
    /// nclass matrix.The result contains predicted probability of each data point belonging to each class.
    /// </summary>
    SoftProb,
}
