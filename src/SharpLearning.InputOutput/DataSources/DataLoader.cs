using System;

namespace SharpLearning.InputOutput.DataSources
{
    public delegate DataBatch<T> Loader<T>(int[] indices);

    public class DataLoader<T>
    {
        readonly Loader<T> m_loader;

        public DataLoader(Loader<T> loader, int sampleCount)
        {
            m_loader = loader ?? throw new ArgumentNullException(nameof(loader));
            SampleCount = sampleCount;
        }

        public int SampleCount { get; }
        public DataBatch<T> Load(int[] indices) => m_loader(indices);
    }
}
