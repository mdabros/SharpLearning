using System;

namespace SharpLearning.XGBoost
{
    /// <summary>
    /// XGBoost regression objectives.
    /// </summary>
    public static class RegressionObjectiveExtensions
    {
        /// <summary>
        /// Convert regression objective to the xgboost parameter string.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string ToXGBoostString(this RegressionObjective type)
        {
            switch (type)
            {
                case RegressionObjective.Linear:
                    return "reg:linear";
                case RegressionObjective.Poisson:
                    return "count:poisson";
                case RegressionObjective.Gamma:
                    return "reg:gamma";
                case RegressionObjective.Tweedie:
                    return "reg:tweedie";
                default:
                    throw new ArgumentException("Unknown regression objective: " + type);
            }
        }
    }
}
