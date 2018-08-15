using System;
using System.Linq;

namespace CntkExtensions
{
    public class MemoryMinibatchSource
    {
        readonly Random m_random;
        readonly int[] m_currentSweepIndeces;

        int m_currentBatchStartIndex = -1;
        int[] m_batchIndeces = Array.Empty<int>();
        
        readonly float[] m_minibatch;
        bool m_randomize;

        Tensor m_observations;
        Tensor m_targets;

        int m_singleObservationDataSize;
        int m_singleTargetDataSize;

        public MemoryMinibatchSource(Tensor observations,
            Tensor targets, 
            int seed,
            bool randomize)
        {
            m_observations = observations ?? throw new ArgumentNullException(nameof(observations));
            m_targets = targets ?? throw new ArgumentNullException(nameof(targets));

            if(observations.SampleCount != targets.SampleCount)
            {
                throw new ArgumentException($"observation samples: {observations.SampleCount}" +
                    $" differs from target samples: {targets.SampleCount}");
            }

            m_singleObservationDataSize = observations.SampleShape
                .Aggregate((d1, d2) => d1 * d2);
            
            m_singleTargetDataSize = targets.SampleShape
                .Aggregate((d1, d2) => d1 * d2); ;

            m_currentSweepIndeces = Enumerable.Range(0, TotalSampleCount).ToArray();
            m_random = new Random(seed);
            m_randomize = randomize;
            m_minibatch = Array.Empty<float>();
        }

        public int TotalSampleCount => m_observations.SampleCount;
        
        public (float[] observations, float[] targets, bool isSweepEnd) GetNextMinibatch(int minibatchSizeInSamples)
        {
            CheckIfNewSweepAndShuffle();

            var batchIndeces = GetBatchIndeces(minibatchSizeInSamples, m_currentBatchStartIndex);

            (var observationsMinibatch, var targetsMiniBatch) = NextBatch();

            m_currentBatchStartIndex += minibatchSizeInSamples;
            
            // Start over if sweep end
            var isSweepEnd = m_currentBatchStartIndex >= m_currentSweepIndeces.Length;
            if (isSweepEnd)
            {
                m_currentBatchStartIndex = -1;
            }

            return (observationsMinibatch, targetsMiniBatch, isSweepEnd);
        }

        private (float[] observations, float[] targets) NextBatch()
        {
            var batchSize = m_batchIndeces.Length;
            var observationsMiniBatch = new float[batchSize * m_singleObservationDataSize];
            var targetMinibatch = new float[batchSize * m_singleTargetDataSize];

            for (int i = 0; i < m_batchIndeces.Length; i++)
            {
                var batchIndex = m_batchIndeces[i];

                var observationStartIndex = batchIndex * m_singleObservationDataSize;
                Array.Copy(m_observations.Data, observationStartIndex, observationsMiniBatch, 
                    i * m_singleObservationDataSize, m_singleObservationDataSize);

                var targetStartIndex = batchIndex * m_singleTargetDataSize;
                Array.Copy(m_targets.Data, targetStartIndex, targetMinibatch, 
                    i * m_singleTargetDataSize, m_singleTargetDataSize);
            }

            return (observationsMiniBatch, targetMinibatch);
        }

        void CheckIfNewSweepAndShuffle()
        {
            if (m_currentBatchStartIndex < 0)
            {
                if (m_randomize)
                {
                    // Start new sweep by shuffling indeces inplace.
                    Shuffle(m_currentSweepIndeces, m_random);
                }
                m_currentBatchStartIndex = 0;
            }
        }

        int[] GetBatchIndeces(int minibatchSizeInSamples, int batchStartIndex)
        {
            if (m_batchIndeces.Length != minibatchSizeInSamples)
            {
                m_batchIndeces = new int[minibatchSizeInSamples];
            }

            for (int i = 0; i < minibatchSizeInSamples; i++)
            {
                // Repeat the start so we can fullfil the requested batch size
                var sweepIndex = (i + batchStartIndex) % m_currentSweepIndeces.Length;
                m_batchIndeces[i] = m_currentSweepIndeces[sweepIndex];
            }
            return m_batchIndeces;
        }

        static void Shuffle<T>(T[] array, Random random)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = random.Next(n);
                --n;
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }
}
