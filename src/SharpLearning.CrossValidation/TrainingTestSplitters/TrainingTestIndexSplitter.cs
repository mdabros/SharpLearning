using System;
using System.Linq;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Creates a set of training and test indices based on the provided targets.
    /// The return values are two arrays of indices which can be used with IIndexedLearners.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class TrainingTestIndexSplitter<T> : ITrainingTestIndexSplitter<T>
    {
        readonly IIndexSampler<T> m_indexSampler;
        readonly double m_trainingPercentage;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shuffler">the type of shuffler provided</param>
        /// <param name="trainingPercentage">What percentage of the indices should go to the training set</param>
        public TrainingTestIndexSplitter(IIndexSampler<T> shuffler, double trainingPercentage)
        {
            m_indexSampler = shuffler ?? throw new ArgumentNullException(nameof(shuffler));
            if (trainingPercentage <= 0.0 || trainingPercentage >= 1.0)
            { throw new ArgumentException("Training percentage must be larger than 0.0 and smaller than 1.0"); }
            m_trainingPercentage = trainingPercentage; 
        }

        /// <summary>
        /// Creates a set of training and test indices based on the provided targets
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        public TrainingTestIndexSplit Split(T[] targets)
        {
            var trainingSampleSize = (int)(m_trainingPercentage * (double)targets.Length);
            trainingSampleSize = trainingSampleSize > 0 ? trainingSampleSize : 1;
            var indices = Enumerable.Range(0, targets.Length).ToArray();

            var trainingIndices = m_indexSampler.Sample(targets, trainingSampleSize, indices);
            var testIndices = indices.Except(trainingIndices)
                .ToArray();

            return new TrainingTestIndexSplit(trainingIndices, 
                testIndices);
        }
    }
}
