using System;

namespace SharpLearning.InputOutput.DataSources
{
    public class DataBatch<T>
    {
        public DataBatch(T[] data, int[] sampleShape, int sampleCount)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            SampleShape = sampleShape ?? throw new ArgumentNullException(nameof(sampleShape));
        }

        public T[] Data { get; }
        public int[] SampleShape { get; }
        public int SampleCount { get; }
    }
}
