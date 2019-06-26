using System;

namespace SharpLearning.XGBoost
{
    /// <summary>
    /// Classification objective extensions
    /// </summary>
    public static class ClassificationObjectiveExtensions
    {
        /// <summary>
        /// Convert classification objective to the xgboost parameter string.
        /// </summary>
        /// <param name="objective"></param>
        /// <returns></returns>
        public static string ToXGBoostString(this ClassificationObjective objective)
        {
            switch (objective)
            {
                case ClassificationObjective.BinaryLogistic:
                    return "binary:logistic";
                case ClassificationObjective.BinaryLogisticRaw:
                    return "binary:logitraw";
                case ClassificationObjective.GPUBinaryLogistic:
                    return "gpu:binary:logistic";
                case ClassificationObjective.GPUBinaryLogisticRaw:
                    return "gpu:binary:logitraw";
                case ClassificationObjective.Softmax:
                    return "multi:softmax";
                case ClassificationObjective.SoftProb:
                    return "multi:softprob";
                default:
                    throw new ArgumentException("Unknown objective: " + objective);
            }
        }
    }
}
