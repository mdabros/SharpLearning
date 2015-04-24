using SharpLearning.Containers.Extensions;

namespace SharpLearning.CrossValidation.Samplers
{
    /// <summary>
    /// Stratified index sampler. Samples. 
    /// Takes a stratified sample of size sampleSize with distributions equal to the input data.
    /// http://en.wikipedia.org/wiki/Stratified_sampling
    /// Returns a set of indices corresponding to the samples chosen. 
    /// </summary>
    /// <typeparam name="T">Returns a set of indices corresponding to the samples chosen. </typeparam>
    public sealed class StratifiedIndexSampler<T> : IIndexSampler<T>
    {
        /// <summary>
        /// Takes a stratified sample of size sampleSize with distributions equal to the input data.
        /// Returns a set of indices corresponding to the samples chosen. 
        /// </summary>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <returns></returns>
        public int[] Sample(T[] data, int sampleSize)
        {
            return data.StratifiedIndexSampling(sampleSize);
        }

        /// <summary>
        /// Takes a stratified sample of size sampleSize with distributions equal to the input data.
        /// http://en.wikipedia.org/wiki/Stratified_sampling
        /// Returns a set of indices corresponding to the samples chosen. 
        /// Only samples within the indices provided in dataIndices
        /// </summary>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <param name="dataIndices"></param>
        /// <returns></returns>
        public int[] Sample(T[] data, int sampleSize, int[] dataIndices)
        {
            return data.StratifiedIndexSampling(sampleSize, dataIndices);
        }
    }
}
