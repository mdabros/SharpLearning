using System;

namespace SharpLearning.DataSource
{
    /// <summary>
    /// Loader delegate for the DataLoader class.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="indices"></param>
    /// <returns></returns>
    public delegate DataBatch<T> Loader<T>(int[] indices);

    /// <summary>
    /// DataLoader class wrapping a loader delegate, 
    /// and exposing the sample count available via the loader.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class DataLoader<T>
    {
        readonly Loader<T> m_loader;

        /// <summary>
        /// DataLoader class wrapping a loader delegate, 
        /// and exposing the sample count available via the loader.
        /// </summary>
        /// <param name="loader"></param>
        /// <param name="totalSampleCount"></param>
        public DataLoader(Loader<T> loader, int totalSampleCount)
        {
            m_loader = loader ?? throw new ArgumentNullException(nameof(loader));
            TotalSampleCount = totalSampleCount;
        }

        /// <summary>
        /// The sample count available from the loader.
        /// </summary>
        public int TotalSampleCount { get; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="indices">Indices to load</param>
        /// <returns></returns>
        public DataBatch<T> Load(int[] indices) => m_loader(indices);
    }
}
