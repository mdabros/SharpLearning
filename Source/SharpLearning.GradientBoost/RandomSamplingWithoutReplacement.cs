using SharpLearning.Containers.Extensions;
using System;

namespace SharpLearning.GradientBoost
{
    public sealed class RandomSamplingWithoutReplacement
    {
        readonly Random m_random;
        readonly double m_percentage;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="percentage">Percentage of full size to sample (0.0 to 1.0)</param>
        public RandomSamplingWithoutReplacement(double percentage)
            : this(percentage, 42)
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="percentage">Percentage of full size to sample (0.0 to 1.0)</param>
        /// <param name="seed">Random seed</param>
        public RandomSamplingWithoutReplacement(double percentage, int seed)
        {
            if (percentage < 0.0 || percentage > 1.0) { throw new ArgumentException("Percentage must be between 0.0 and 1.0"); }
            m_percentage = percentage;
            m_random = new Random(seed);
        }

        /// <summary>
        /// Random sampling using Fischer-Yates shuffle:
        /// http://en.wikipedia.org/wiki/Fisher-Yates_shuffle
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="outIndices"></param>
        public void Sample(int[] indices, ref int[] outIndices)
        {
            Array.Copy(indices, outIndices, indices.Length);
            outIndices.Shuffle(m_random);
            var newLength = (int)(indices.Length * m_percentage);

            Array.Resize(ref outIndices, newLength);
        }
    }
}
