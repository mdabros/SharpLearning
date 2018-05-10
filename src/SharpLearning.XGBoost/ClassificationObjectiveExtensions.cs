using System;
using System.Collections.Generic;
using System.Text;

namespace SharpLearning.XGBoost
{
    /// <summary>
    /// XGBoost regression objectives.
    /// </summary>
    public static class ClassificationObjectiveExtensions
    {
        /// <summary>
        /// Convert regression objective to the xgboost parameter string.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string ToXGBoostString(this ClassificationObjective type)
        {
            switch (type)
            {
                case ClassificationObjective.BinaryLogistic:
                    return "binary:logistic";
                case ClassificationObjective.BinaryLogisticRaw:
                    return "binary:logitraw";
                case ClassificationObjective.Softmax:
                    return "multi:softmax";
                default:
                    throw new ArgumentException("Unknown classification objective: " + type);
            }
        }
    }
}
