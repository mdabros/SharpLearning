using System;

namespace SharpLearning.XGBoost;

/// <summary>
/// 
/// </summary>
public static class BoosterTypeExtensions
{
    /// <summary>
    /// Convert booster type to the xgboost parameter string.
    /// </summary>
    /// <param name="type"></param>
    /// <returns></returns>
    public static string ToXGBoostString(this BoosterType type)
    {
        switch (type)
        {
            case BoosterType.GBTree:
                return "gbtree";
            case BoosterType.GBLinear:
                return "gblinear";
            case BoosterType.DART:
                return "dart";
            default:
                throw new ArgumentException("Unknown BoosterType: " + type);
        }
    }
}
