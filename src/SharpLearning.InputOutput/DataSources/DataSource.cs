using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.InputOutput.DataSources
{
    /// <summary>
    /// Data source for serving batches of data.
    /// This is convenient when data sets becomes to large to fit in memory.
    /// Typically this will be used with algorithms that supports streaming data, 
    /// like stochastic gradient descent.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class DataSource<T>
    {
        readonly IReadOnlyDictionary<string, DataLoader<T>> m_idToDataLoader;
        readonly int m_batchSize;
        readonly bool m_shuffle;

        readonly Random m_random;
        readonly int[] m_currentSweepIndices;

        int m_currentBatchStartIndex = -1;
        int[] m_batchIndices = Array.Empty<int>();

        /// <summary>
        /// Data source for serving batches of data.
        /// </summary>
        /// <param name="idToDataLoader">Map from string id to data loader. 
        /// For instance, a separate loader could be used for loading the features/observations, and the targets.</param>
        /// <param name="batchSize">Sample count in each batch.</param>
        /// <param name="shuffle">Randomize the order the data is loaded in.</param>
        /// <param name="seed">Seed for randomization if enabled.</param>
        public DataSource(IReadOnlyDictionary<string, DataLoader<T>> idToDataLoader,
            int batchSize, bool shuffle, int seed)
        {
            m_idToDataLoader = idToDataLoader ?? throw new ArgumentNullException(nameof(idToDataLoader));
            m_batchSize = batchSize;
            m_shuffle = shuffle;
            m_random = new Random(seed);

            var sampleCount = m_idToDataLoader.Values.First().SampleCount;
            foreach (var loader in m_idToDataLoader.Values)
            {
                if(loader.SampleCount != sampleCount)
                {
                    throw new ArgumentException("");
                }
            }

            m_currentSweepIndices = Enumerable.Range(0, sampleCount).ToArray();
        }

        /// <summary>
        /// Gets the next batch of data.
        /// Returns a dictionary mapping from id to the corresponding data batch.
        /// </summary>
        /// <returns></returns>
        public (IReadOnlyDictionary<string, DataBatch<T>>, bool isSweepEnd) GetNextBatch()
        {
            CheckIfNewSweepAndShuffle();
            UpdateBatchIndices(m_batchSize, m_currentBatchStartIndex);
            var batch = NextBatch();

            m_currentBatchStartIndex += m_batchSize;

            // Start over if sweep end
            var isSweepEnd = m_currentBatchStartIndex >= m_currentSweepIndices.Length;
            if (isSweepEnd)
            {
                m_currentBatchStartIndex = -1;
            }

            return (batch, isSweepEnd);
        }

        Dictionary<string, DataBatch<T>> NextBatch()
        {
            var batch = new Dictionary<string, DataBatch<T>>();
            foreach (var idToLoader in m_idToDataLoader)
            {
                var id = idToLoader.Key;
                var loader = idToLoader.Value;
                var data = loader.Load(m_batchIndices);

                batch.Add(id, data);
            }

            return batch;
        }

        void CheckIfNewSweepAndShuffle()
        {
            if (m_currentBatchStartIndex < 0)
            {
                if (m_shuffle)
                {
                    // Start new sweep by shuffling indexes in place.
                    Shuffle(m_currentSweepIndices, m_random);
                }
                m_currentBatchStartIndex = 0;
            }
        }

        void UpdateBatchIndices(int minibatchSizeInSamples, int batchStartIndex)
        {
            if (m_batchIndices.Length != minibatchSizeInSamples)
            {
                m_batchIndices = new int[minibatchSizeInSamples];
            }

            for (int i = 0; i < minibatchSizeInSamples; i++)
            {
                // Repeat the start so we can fulfill the requested batch size
                var sweepIndex = (i + batchStartIndex) % m_currentSweepIndices.Length;
                m_batchIndices[i] = m_currentSweepIndices[sweepIndex];
            }
        }

        static void Shuffle<TIndex>(TIndex[] array, Random random)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = random.Next(n);
                --n;
                TIndex temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}
