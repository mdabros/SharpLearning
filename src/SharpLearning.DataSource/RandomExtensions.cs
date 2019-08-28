using System;

namespace SharpLearning.DataSource
{
    public static class RandomExtensions
    {
        public static float Sample(this Random random, float max)
        {
            return Sample(random, 0, max);
        }

        public static float Sample(this Random random, float min, float max)
        {
            return (float)random.NextDouble() * (max - min) + min;
        }
    }
}
