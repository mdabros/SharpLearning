using System;

namespace SharpLearning.DataSource
{
    /// <summary>
    /// Container for batch data.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class DataBatch<T>
    {
        /// <summary>
        /// Container for batch data.
        /// </summary>
        /// <param name="data"></param>
        /// <param name="sampleShape"></param>
        /// <param name="sampleCount"></param>
        public DataBatch(T[] data, int[] sampleShape, int sampleCount)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            SampleShape = sampleShape ?? throw new ArgumentNullException(nameof(sampleShape));
            SampleCount = sampleCount;
        }

        /// <summary>
        /// All samples from in the batch are stored in a single array.
        /// </summary>
        public T[] Data { get; }

        /// <summary>
        /// Shape of each sample.
        /// </summary>
        public int[] SampleShape { get; }

        /// <summary>
        /// Sample count in the batch.
        /// </summary>
        public int SampleCount { get; }
    }
}
