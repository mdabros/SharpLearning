namespace SharpLearning.Optimization.ParameterSamplers
{
    /// <summary>
    /// Defines the interface for a parameter samplers. 
    /// </summary>
    public interface IParameterSampler
    {
        /// <summary>
        /// Returns a sample within in the specidied min/max boundaries.
        /// </summary>
        /// <param name="min">Minimum bound</param>
        /// <param name="max">Maximum bound</param>
        /// <param name="random"></param>
        /// <returns></returns>
        double Sample(double min, double max);
    }
}
