using System;
using System.Linq;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.CrossValidation.Samplers
{
    /// <summary>
    /// Random index sampler. Takes at random a sample of size sample size
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class RandomIndexSampler<T> : IIndexSampler<T>
    {
        readonly Random m_random;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="seed"></param>
        public RandomIndexSampler(int seed)
        {
            m_random = new Random(seed);
        }

        /// <summary>
        /// 
        /// </summary>
        public RandomIndexSampler()
            : this(42)
        {
        }

        /// <summary>
        /// Random index sampler. Takes at random a sample of size sample size
        /// </summary>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <returns></returns>
        public int[] Sample(T[] data, int sampleSize)
        {
            if (data.Length < sampleSize)
            {
                throw new ArgumentException("Sample size " + sampleSize + 
                    " is larger than data size " + data.Length);
            }

            var indices = Enumerable.Range(0, data.Length).ToArray();
            indices.Shuffle(m_random);
            
            return indices.Take(sampleSize).ToArray();
        }

        /// <summary>
        /// Random index sampler. Takes at random a sample of size sample size. 
        /// Only samples within the indices provided in dataIndices
        /// </summary>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <param name="dataIndices"></param>
        /// <returns></returns>
        public int[] Sample(T[] data, int sampleSize, int[] dataIndices)
        {
            if (data.Length < sampleSize) 
            {
                throw new ArgumentException("Sample size " + sampleSize + 
                    " is larger than data size " + data.Length);
            }

            if (data.Length < dataIndices.Length) 
            {
                throw new ArgumentException("dataIndice size " + dataIndices.Length + 
                    " is larger than data size " + data.Length);
            }

            var indices = dataIndices.ToArray();
            indices.Shuffle(m_random);

            return indices.Take(sampleSize).ToArray();
        }
    }
}
