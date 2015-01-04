using SharpLearning.CrossValidation.Shufflers;
using System;

namespace SharpLearning.CrossValidation.TrainingValidationSplitters
{
    /// <summary>
    /// Creates a set of training and validation indices based on the provided targets.
    /// The indices are randomly shuffled before the split.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class RandomTrainingValidationIndexSplitter<T> : TrainingValidationIndexSplitter<T>
    {
        /// <summary>
        /// Creates a set of training and validation indices based on the provided targets.
        /// The indices are randomly shuffled before the split.
        /// </summary>
        /// <param name="trainingPercentage"></param>
        public RandomTrainingValidationIndexSplitter(double trainingPercentage)
            : this(trainingPercentage, DateTime.Now.Millisecond)
        {
        }

        /// <summary>
        /// Creates a set of training and validation indices based on the provided targets.
        /// The indices are randomly shuffled before the split.
        /// </summary>
        /// <param name="trainingPercentage"></param>
        /// <param name="seed"></param>
        public RandomTrainingValidationIndexSplitter(double trainingPercentage, int seed)
            : base(new RandomCrossValidationShuffler<T>(seed), trainingPercentage)
        {
        }
    }
}
