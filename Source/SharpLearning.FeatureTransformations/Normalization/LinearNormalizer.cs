
namespace SharpLearning.FeatureTransformations.Normalization
{
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
        public double Normalize(double newMin, double newMax, double oldMin, double oldMax, double value)
        {
            if (value == oldMin)
                value = newMin;
            else if (value == oldMax)
                value = newMax;
            else
                value = newMin + (newMax - newMin) *
                (value - oldMin) / (oldMax - oldMin);
            return value;
        }
    }
}
