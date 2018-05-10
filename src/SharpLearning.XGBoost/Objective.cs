namespace SharpLearning.XGBoost
{
    /// <summary>
    /// XGBoost objectives.
    /// </summary>
    public enum Objective
    {
        /// <summary>
        /// linear regression.
        /// </summary>
        LinearRegression,

        /// <summary>
        /// logistic regression.
        /// </summary>
        LogisticRegression,

        /// <summary>
        /// logistic regression for binary classification, output probability.
        /// </summary>
        BinaryLogistic,

        /// <summary>
        /// logistic regression for binary classification, output score before logistic transformation.
        /// </summary>
        BinaryLogisticRaw,

        /// <summary>
        /// GPU version of linear regression evaluated on the GPU,
        /// note that like the GPU histogram algorithm, 
        /// they can only be used when the entire training session uses the same dataset.
        /// </summary>
        GPULinear,

        /// <summary>
        /// GPU version of logistic regression evaluated on the GPU,
        /// note that like the GPU histogram algorithm, 
        /// they can only be used when the entire training session uses the same dataset.
        /// </summary>
        GPULogistic,

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
        /// poisson regression for count data, output mean of poisson distribution,
        /// max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization).
        /// </summary>
        CountPoisson,

        /// <summary>
        /// Cox regression for right censored survival time data (negative values are considered right censored). 
        /// Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) 
        /// in the proportional hazard function h(t) = h0(t) * HR).
        /// </summary>
        SurvivalCox,

        /// <summary>
        /// Multiclass classification using the softmax objective.
        /// </summary>
        Softmax,

        /// <summary>
        /// same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, 
        /// nclass matrix.The result contains predicted probability of each data point belonging to each class.
        /// </summary>
        SoftProb,

        /// <summary>
        /// ranking task by minimizing the pairwise loss.
        /// </summary>
        RankPairwise,

        /// <summary>
        /// gamma regression with log-link.Output is a mean of gamma distribution.
        /// It might be useful, e.g., for modeling insurance claims severity, 
        /// or for any outcome that might be gamma-distributed
        /// </summary>
        GammaRegression,

        /// <summary>
        /// Tweedie regression with log-link.It might be useful, e.g., 
        /// for modeling total loss in insurance, 
        /// or for any outcome that might be Tweedie-distributed.
        /// </summary>
        TweedieRegression
    }
}
