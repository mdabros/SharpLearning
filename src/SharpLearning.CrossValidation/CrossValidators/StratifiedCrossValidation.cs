using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.CrossValidators
{
    /// <summary>
    /// Uses stratified sampling to shuffle the indices for cross validation
    /// http://en.wikipedia.org/wiki/Stratified_sampling
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public sealed class StratifiedCrossValidation<TPrediction> : CrossValidation<TPrediction>
    {
        /// <summary>
        /// Cross validation for evaluating how learning algorithms generalize on new data
        /// </summary>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        /// <param name="seed"></param>
        public StratifiedCrossValidation(int crossValidationFolds, int seed = 42)
            : base(new StratifiedIndexSampler<double>(seed), crossValidationFolds)
        {
        }
    }
}
