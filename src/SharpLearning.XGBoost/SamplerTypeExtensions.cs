using System;

namespace SharpLearning.XGBoost;

/// <summary>
/// 
/// </summary>
public static class SamplerTypeExtensions
{
    /// <summary>
    /// Convert sampler type to the xgboost parameter string.
    /// </summary>
    /// <param name="type"></param>
    /// <returns></returns>
    public static string ToXGBoostString(this SamplerType type)
    {
        switch (type)
        {
            case SamplerType.Uniform:
                return "uniform";
            case SamplerType.Weighted:
                return "weighted";
            default:
                throw new ArgumentException("Unknown sampler type: " + type); ;
        }
    }
}
