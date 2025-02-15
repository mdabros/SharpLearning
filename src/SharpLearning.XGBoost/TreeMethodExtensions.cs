using System;

namespace SharpLearning.XGBoost;

/// <summary>
///
/// </summary>
public static class TreeMethodExtensions
{
    /// <summary>
    /// Convert regression objective to the xgboost parameter string.
    /// </summary>
    /// <param name="type"></param>
    /// <returns></returns>
    public static string ToXGBoostString(this TreeMethod type)
    {
        switch (type)
        {
            case TreeMethod.Auto:
                return "auto";
            case TreeMethod.Exact:
                return "exact";
            case TreeMethod.Approx:
                return "approx";
            case TreeMethod.Hist:
                return "hist";
            case TreeMethod.GPUExact:
                return "gpu_exact";
            case TreeMethod.GPUHist:
                return "gpu_hist";
            default:
                throw new ArgumentException("Unknown TreeMethod type: " + type);
        }
    }
}
