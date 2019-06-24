using System;
using System.Linq;

namespace SharpLearning.CrossValidation.Samplers
{
    /// <summary>
    /// No shuffle index sampler. 
    /// Simply takes the amount of samples specified by sample size from the start of the data
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class NoShuffleIndexSampler<T> : IIndexSampler<T>
    {
        /// <summary>
        /// Simply takes the amount of samples specified by sample size from the start of the data
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
            
            return Enumerable.Range(0, sampleSize).ToArray();
        }

        /// <summary>
        /// Simply takes the amount of samples specified by sample size from the provided dataIndices
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
            
            return dataIndices.Take(sampleSize).ToArray();
        }
    }
}
