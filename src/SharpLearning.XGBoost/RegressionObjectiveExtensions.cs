using System;

namespace SharpLearning.XGBoost;

/// <summary>
/// Regression objective extensions
/// </summary>
public static class RegressionObjectiveExtensions
{
    /// <summary>
    /// Convert regression objective to the xgboost parameter string.
    /// </summary>
    /// <param name="objective"></param>
    /// <returns></returns>
    public static string ToXGBoostString(this RegressionObjective objective)
    {
        switch (objective)
        {
            case RegressionObjective.LinearRegression:
                return "reg:linear";
            case RegressionObjective.LogisticRegression:
                return "reg:logistic";
            case RegressionObjective.GPULinear:
                return "gpu:reg:linear";
            case RegressionObjective.GPULogistic:
                return "gpu:reg:logistic";
            case RegressionObjective.CountPoisson:
                return "count:poisson";
            case RegressionObjective.SurvivalCox:
                return "survival:cox";
            case RegressionObjective.RankPairwise:
                return "rank:pairwise";
            case RegressionObjective.GammaRegression:
                return "reg:gamma";
            case RegressionObjective.TweedieRegression:
                return "reg:tweedie";
            default:
                throw new ArgumentException("Unknown objective: " + objective);
        }
    }
}
