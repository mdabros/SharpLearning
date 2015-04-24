using SharpLearning.CrossValidation.Samplers;
using System;

namespace SharpLearning.CrossValidation.TrainingValidationSplitters
{
    /// <summary>
    /// Creates a set of training and validation indices based on the provided targets.
    /// The indices are stratified before the split. This ensure that the distributions of training set and 
    /// validation set are equal or at least very similar. 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class StratifiedTrainingValidationIndexSplitter<T> : TrainingValidationIndexSplitter<T>
    {
        /// <summary>
        /// The indices are stratified before the split. This ensure that the distributions of training set and 
        /// validation set are equal or at least very similar. 
        /// </summary>
        /// <param name="trainingPercentage"></param>
        public StratifiedTrainingValidationIndexSplitter(double trainingPercentage)
            : base(new StratifiedIndexSampler<T>(), trainingPercentage)
        {
        }
    }
}
