namespace SharpLearning.XGBoost
{
    /// <summary>
    /// Regression objectives.
    /// </summary>
    public enum RegressionObjective
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
