namespace SharpLearning.XGBoost
{
    public enum RegressionObjective
    {
        /// <summary>
        /// Linear regression objective.
        /// </summary>
        Linear,

        /// <summary>
        /// Poisson regression for count data, output mean of poisson distribution.
        /// </summary>
        Poisson,

        /// <summary>
        /// Gamma regression with log-link. Output is a mean of gamma distribution. 
        /// It might be useful, e.g., for modeling insurance claims severity, 
        /// or for any outcome that might be gamma-distributed.
        /// </summary>
        Gamma,

        /// <summary>
        /// Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, 
        /// or for any outcome that might be Tweedie-distributed.
        /// </summary>
        Tweedie
    }
}
