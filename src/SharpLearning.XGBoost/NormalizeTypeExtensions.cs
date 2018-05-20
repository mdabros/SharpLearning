using System;

namespace SharpLearning.XGBoost
{
    /// <summary>
    /// 
    /// </summary>
    public static class NormalizeTypeExtensions
    {
        /// <summary>
        /// Convert normalize type to the xgboost parameter string.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string ToXGBoostString(this NormalizeType type)
        {
            switch (type)
            {
                case NormalizeType.Tree:
                    return "tree";
                case NormalizeType.Forest:
                    return "forest";
                default:
                    throw new ArgumentException("Unknown normalize type: " + type);
            }
        }
    }
}
