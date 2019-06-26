using System;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Creates a set of training and test indices based on the provided targets.
    /// The indices are randomly shuffled before the split.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class RandomTrainingTestIndexSplitter<T> : TrainingTestIndexSplitter<T>
    {
        /// <summary>
        /// Creates a set of training and test indices based on the provided targets.
        /// The indices are randomly shuffled before the split.
        /// </summary>
        /// <param name="trainingPercentage"></param>
        public RandomTrainingTestIndexSplitter(double trainingPercentage)
            : this(trainingPercentage, DateTime.Now.Millisecond)
        {
        }

        /// <summary>
        /// Creates a set of training and test indices based on the provided targets.
        /// The indices are randomly shuffled before the split.
        /// </summary>
        /// <param name="trainingPercentage"></param>
        /// <param name="seed"></param>
        public RandomTrainingTestIndexSplitter(double trainingPercentage, int seed)
            : base(new RandomIndexSampler<T>(seed), trainingPercentage)
        {
        }
    }
}
