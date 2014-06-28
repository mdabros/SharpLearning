using System;
using SharpLearning.Containers;

namespace SharpLearning.CrossValidation.Shufflers
{
    /// <summary>
    /// uses a random strategy to shuffles the provided indices
    /// </summary>
    public sealed class RandomCrossValidationShuffler<T> : ICrossValidationShuffler<T>
    {
        readonly Random m_random;

        public RandomCrossValidationShuffler(int seed)
        {
            m_random = new Random(seed);
        }

        /// <summary>
        /// Uses a random strategy to shuffles the provided indices.
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="targets"></param>
        /// <param name="folds"></param>
        public void Shuffle(int[] indices, T[] targets, int folds)
        {
            indices.Shuffle(m_random);
        }
    }
}
