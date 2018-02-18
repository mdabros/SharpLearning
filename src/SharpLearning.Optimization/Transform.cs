namespace SharpLearning.Optimization
{
    /// <summary>
    /// Enum for specifying predefined parameter samplers
    /// </summary>
    public enum Transform
    {
        /// <summary>
        /// Linear scale. For smaller ranges like min: 64 and max: 256.
        /// </summary>
        Linear,

        /// <summary>
        /// Logarithmic scale. For larger ranges like min: 0.0001 and max: 1.0.
        /// </summary>
        Logarithmic
    }
}
