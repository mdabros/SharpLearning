using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Creates a set of training and test indices based on the provided targets.
    /// The indices are not shuffled before the split keeping the order of the data.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class NoShuffleTrainingTestIndexSplitter<T> : TrainingTestIndexSplitter<T>
    {
        /// <summary>
        /// Creates a set of training and test indices based on the provided targets.
        /// The indices are not shuffled before the split keeping the order of the data.
        /// </summary>
        /// <param name="trainingPercentage"></param>
        public NoShuffleTrainingTestIndexSplitter(double trainingPercentage)
            : base(new NoShuffleIndexSampler<T>(), trainingPercentage)
        {
        }
    }
}
