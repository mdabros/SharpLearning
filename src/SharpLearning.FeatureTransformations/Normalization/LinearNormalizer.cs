
namespace SharpLearning.FeatureTransformations.Normalization;

/// <summary>
/// Normalizes a value from one (oldMin, oldMax) range to a new range (newMax, newMin)
/// </summary>
public sealed class LinearNormalizer
{
    /// <summary>
    /// Normalizes a value from one (oldMin, oldMax) range to a new range (newMax, newMin)
    /// </summary>
    /// <param name="newMin"></param>
    /// <param name="newMax"></param>
    /// <param name="oldMin"></param>
    /// <param name="oldMax"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    public static double Normalize(double newMin, double newMax, double oldMin, double oldMax, double value)
    {
        if (value == oldMin)
        {
            return newMin;
        }
        else if (value == oldMax)
        {
            return newMax;
        }
        else
        {
            return newMin + (newMax - newMin) * (value - oldMin) / (oldMax - oldMin);
        }
    }
}
