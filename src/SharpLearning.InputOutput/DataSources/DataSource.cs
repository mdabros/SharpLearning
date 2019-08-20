using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.InputOutput.DataSources
{
    public class DataSource<T>
    {
        readonly IReadOnlyDictionary<string, DataLoader<T>> m_idToDataLoader;
        readonly int m_batchSize;
        readonly bool m_shuffle;

        readonly Random m_random;
        readonly int[] m_currentSweepIndeces;

        int m_currentBatchStartIndex = -1;
        int[] m_batchIndeces = Array.Empty<int>();

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
        }

        public (IReadOnlyDictionary<string, DataBatch<T>>, bool isSweepEnd) GetNextBatch()
        {
            CheckIfNewSweepAndShuffle();
            UpdateBatchIndeces(m_batchSize, m_currentBatchStartIndex);

            var batch = NextBatch();

            m_currentBatchStartIndex += m_batchSize;

            // Start over if sweep end
            var isSweepEnd = m_currentBatchStartIndex >= m_currentSweepIndeces.Length;
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
                var data = loader.Load(m_batchIndeces);

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
                    Shuffle(m_currentSweepIndeces, m_random);
                }
                m_currentBatchStartIndex = 0;
            }
        }

        void UpdateBatchIndeces(int minibatchSizeInSamples, int batchStartIndex)
        {
            if (m_batchIndeces.Length != minibatchSizeInSamples)
            {
                m_batchIndeces = new int[minibatchSizeInSamples];
            }

            for (int i = 0; i < minibatchSizeInSamples; i++)
            {
                // Repeat the start so we can fulfill the requested batch size
                var sweepIndex = (i + batchStartIndex) % m_currentSweepIndeces.Length;
                m_batchIndeces[i] = m_currentSweepIndeces[sweepIndex];
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
