using System;

namespace SharpLearning.XGBoost
{
    /// <summary>
    /// Objective extensions
    /// </summary>
    public static class ObjectiveExtensions
    {
        /// <summary>
        /// Convert objective to the xgboost parameter string.
        /// </summary>
        /// <param name="objective"></param>
        /// <returns></returns>
        public static string ToXGBoostString(this Objective objective)
        {
            switch (objective)
            {
                case Objective.LinearRegression:
                    return "reg:linear";
                case Objective.LogisticRegression:
                    return "reg: logistic";
                case Objective.BinaryLogistic:
                    return "binary:logistic";
                case Objective.BinaryLogisticRaw:
                    return "binary:logitraw";
                case Objective.GPULinear:
                    return "gpu:reg:linear";
                case Objective.GPULogistic:
                    return "gpu:reg:logistic";
                case Objective.GPUBinaryLogistic:
                    return "gpu:binary:logistic";
                case Objective.GPUBinaryLogisticRaw:
                    return "gpu:binary:logitraw";
                case Objective.CountPoisson:
                    return "count:poisson";
                case Objective.SurvivalCox:
                    return "survival:cox";
                case Objective.Softmax:
                    return "multi:softmax";
                case Objective.SoftProb:
                    return "multi:softprob";
                case Objective.RankPairwise:
                    return "rank:pairwise";
                case Objective.GammaRegression:
                    return "reg:gamma";
                case Objective.TweedieRegression:
                    return "reg:tweedie";
                default:
                    throw new ArgumentException("Unknown objective: " + objective);
            }
        }
    }
}
