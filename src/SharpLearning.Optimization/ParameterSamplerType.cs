namespace SharpLearning.Optimization
{
    /// <summary>
    /// Enum for specifying predefined parameter samplers
    /// </summary>
    public enum ParameterSamplerType
    {
        /// <summary>
        /// Linear. Samples random uniform on the linear scale.
        /// For smaller ranges like min: 64 and max: 256.
        /// </summary>
        RandomUniformLinear,

        /// <summary>
        /// Logarithmic. Samples random uniform on the logarithmic scale.
        /// For larger ranges like min: 0.0001 and max: 1.0.
        /// </summary>
        RandomUniformLogarithmic
    }
}
