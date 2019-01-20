namespace SharpLearning.Optimization
{
    /// <summary>
    /// Enum for specifying predefined parameter samplers
    /// </summary>
    public enum Transform
    {
        /// <summary>
        /// Linear scale. For ranges with a small difference in numerical scale, like min: 64 and max: 256.
        /// </summary>
        Linear,

        /// <summary>
        /// Logarithmic scale. For ranges with a large difference in numerical scale, like min: 0.0001 and max: 1.0.
        /// </summary>
        Log10,

        /// <summary>
        /// ExponentialAverage scale. For ranges close to one, like min: 0.9 and max: 0.999.
        /// </summary>
        ExponentialAverage
    }
}
