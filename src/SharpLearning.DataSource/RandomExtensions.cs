using System;

namespace SharpLearning.DataSource
{
    /// <summary>
    /// Extensions for the random class.
    /// </summary>
    public static class RandomExtensions
    {
        /// <summary>
        /// Samples uniform between min and max.
        /// </summary>
        /// <param name="random"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        public static float SampleUniform(this Random random, float min, float max)
        {
            return (float)random.NextDouble() * (max - min) + min;
        }
    }
}
