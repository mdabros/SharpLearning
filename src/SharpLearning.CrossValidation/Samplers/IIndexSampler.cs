
namespace SharpLearning.CrossValidation.Samplers
{
    /// <summary>
    /// Interface for index sampler.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IIndexSampler<T>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <returns></returns>
        int[] Sample(T[] data, int sampleSize);
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="data"></param>
        /// <param name="sampleSize"></param>
        /// <param name="dataIndices"></param>
        /// <returns></returns>
        int[] Sample(T[] data, int sampleSize, int[] dataIndices);
    }
}
