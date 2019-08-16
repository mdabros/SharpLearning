using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.TrainingTestSplitters
{
    /// <summary>
    /// Creates a set of training and test indices based on the provided targets.
    /// The indices are stratified before the split. This ensure that the distributions of training set and 
    /// test set are equal or at least very similar. 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public sealed class StratifiedTrainingTestIndexSplitter<T> : TrainingTestIndexSplitter<T>
    {
        /// <summary>
        /// The indices are stratified before the split. This ensure that the distributions of training set and 
        /// test set are equal or at least very similar. 
        /// </summary>
        /// <param name="trainingPercentage"></param>
        /// <param name="seed"></param>
        public StratifiedTrainingTestIndexSplitter(double trainingPercentage, int seed = 42)
            : base(new StratifiedIndexSampler<T>(seed), trainingPercentage)
        {
        }
    }
}
