using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.TrainingValidationSplitters
{
    /// <summary>
    /// Creates a set of training and validation indices based on the provided targets.
    /// The indices are not shuffled before the split keeping the order of the data.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class NoShuffleTrainingValidationIndexSplitter<T> : TrainingValidationIndexSplitter<T>
    {
        /// <summary>
        /// Creates a set of training and validation indices based on the provided targets.
        /// The indices are not shuffled before the split keeping the order of the data.
        /// </summary>
        /// <param name="trainingPercentage"></param>
        public NoShuffleTrainingValidationIndexSplitter(double trainingPercentage)
            : base(new NoShuffleIndexSampler<T>(), trainingPercentage)
        {
        }
    }
}
