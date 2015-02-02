using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using System;

namespace SharpLearning.CrossValidation.Shufflers
{
    /// <summary>
    /// Uses stratified sampling to shuffle the indices for cross validation
    /// http://en.wikipedia.org/wiki/Stratified_sampling
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class StratifyCrossValidationShuffler<T> : ICrossValidationShuffler<T>
    {
        readonly Random m_random;

        public StratifyCrossValidationShuffler()
            : this(42)
        {
        }

        public StratifyCrossValidationShuffler(int seed)
        {
            m_random = new Random(seed);
        }

        /// <summary>
        /// Uses stratified sampling to shuffle the indices for cross validation
        /// </summary>
        /// <param name="indices"></param>
        /// <param name="targets"></param>
        /// <param name="folds"></param>
        public void Shuffle(int[] indices, T[] targets, int folds)
        {
            indices.Stratify(targets, m_random, folds);
        }
    }
}
