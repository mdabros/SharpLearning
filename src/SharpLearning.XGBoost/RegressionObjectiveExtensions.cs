using System;

namespace SharpLearning.XGBoost
{
    public static class RegressionObjectiveExtensions
    {
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
                    throw new ArgumentException("Unknown regression type: " + type);
            }
        }
    }
}
