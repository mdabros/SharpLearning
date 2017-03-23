using System;
using System.Linq;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Tensors;

namespace SharpLearning.Neural.LayersNew
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class Batcher
    {
        int[] m_indices;
        Random m_random;
        int m_currentIndex;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="seed"></param>
        public void Initialize(TensorShape observations, int seed)
        {
            m_indices = Enumerable.Range(0, observations.Dimensions[0]).ToArray();
            m_random = new Random(seed);
            Shuffle();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="batchSize"></param>
        /// <param name="net"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="batchObservations"></param>
        /// <param name="BatchTargets"></param>
        public bool Next(int batchSize, NeuralNet2 net,
            Tensor<double> observations, Tensor<double> targets,
            Tensor<double> batchObservations, Tensor<double> BatchTargets)
        {
            if(m_currentIndex + batchSize >= observations.Dimensions[0])
            {
                return false;
            }

            var batchIndices = new int[batchSize];
            Array.Copy(m_indices, m_currentIndex, batchIndices, 0, batchSize);

            observations.SliceCopy(batchIndices, batchObservations);
            targets.SliceCopy(batchIndices, BatchTargets);
            
            m_currentIndex += batchSize;

            net.SetNextBatch(batchObservations, BatchTargets);

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        public void Shuffle()
        {
            m_indices.Shuffle(m_random);
            m_currentIndex = 0;
        }
    }
}
