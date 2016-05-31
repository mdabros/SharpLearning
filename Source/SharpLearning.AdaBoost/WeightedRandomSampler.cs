using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using System;
using System.Linq;

namespace SharpLearning.AdaBoost
{
    /// <summary>
    /// Weighted sampling with replacement based on:
    /// http://stackoverflow.com/questions/2140787/select-random-k-elements-from-a-list-whose-elements-have-weights/2149533#2149533
    /// The algorithm should be O(n+m) where m are the number of items and n is the number of samples.
    /// </summary>
    public sealed class WeightedRandomSampler
    {
        readonly Random m_random;

        /// <summary>
        /// Weighted sampling with replacement based on:
        /// http://stackoverflow.com/questions/2140787/select-random-k-elements-from-a-list-whose-elements-have-weights/2149533#2149533
        /// The algorithm should be O(n+m) where m are the number of items and n is the number of samples.
        /// </summary>
        public WeightedRandomSampler()
            : this(42)
        {
        }

        /// <summary>
        /// Weighted sampling with replacement based on:
        /// http://stackoverflow.com/questions/2140787/select-random-k-elements-from-a-list-whose-elements-have-weights/2149533#2149533
        /// The algorithm should be O(n+m) where m are the number of items and n is the number of samples.
        /// </summary>
        /// <param name="seed"></param>
        public WeightedRandomSampler(int seed)
        {
            m_random = new Random(seed);
        }

        /// <summary>
        /// Weighted sampling with replacement based on:
        /// http://stackoverflow.com/questions/2140787/select-random-k-elements-from-a-list-whose-elements-have-weights/2149533#2149533
        /// The algorithm should be O(n+m) where m are the number of items and n is the number of samples.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="weights"></param>
        /// <param name="outIndices"></param>
        public void Sample(int[] indices, double[] weights, int[] outIndices)
        {
            var totalWeight = weights.Sum(indices);
            var i = 0;

            var index = indices.First();
            var weight = weights[index];

            var samples = outIndices.Length;
            var current = 0;

            while (samples > 0)
            {
                var x = totalWeight * (1.0 - Math.Pow(m_random.NextDouble(), (1.0 / samples)));
                totalWeight -= x;
                while (x > weight)
                {
                    x -= weight;
                    i += 1;
                    index = indices[i];
                    weight = weights[index];
                }
                weight -= x;
                outIndices[current++] = index;
                samples -= 1;
            }
        }
    }
}
