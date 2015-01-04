using SharpLearning.CrossValidation.Shufflers;
using System;
using System.Linq;

namespace SharpLearning.CrossValidation.TrainingValidationSplitters
{
    /// <summary>
    /// Creates a set of training and validation indices based on the provided targets.
    /// The return values are two arrays of indices which can be used with IIndexedLearners.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class TrainingValidationIndexSplitter<T>
    {
        readonly ICrossValidationShuffler<T> m_shuffler;
        readonly double m_trainingPercentage;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shuffler">the type of shuffler provided</param>
        /// <param name="trainingPercentage">What percentage of the indices should go to the training set</param>
        public TrainingValidationIndexSplitter(ICrossValidationShuffler<T> shuffler, double trainingPercentage)
        {
            if (shuffler == null) { throw new ArgumentNullException("shuffler"); }
            if (trainingPercentage <= 0.0 || trainingPercentage >= 1.0)
            { throw new ArgumentException("Training percentage must be larger than 0.0 and smaller than 1.0"); }
            m_shuffler = shuffler;
            m_trainingPercentage = trainingPercentage; 
        }

        /// <summary>
        /// Creates a set of training and validation indices based on the provided targets
        /// </summary>
        /// <param name="targets"></param>
        /// <returns></returns>
        public TrainingValidationIndexSplit Split(T[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            var folds = (int)Math.Round(1.0 / (1.0 - m_trainingPercentage));
            m_shuffler.Shuffle(indices, targets, folds);

            var trainingSampleSize = (int)(m_trainingPercentage * (double)indices.Length);
            trainingSampleSize = trainingSampleSize > 0 ? trainingSampleSize : 1;

            var trainingIndices = indices.Take(trainingSampleSize)
                .ToArray();
            var validationIndices = indices.Except(trainingIndices)
                .ToArray();

            return new TrainingValidationIndexSplit(trainingIndices, 
                validationIndices);
        }
    }
}
