using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.CrossValidators
{
    /// <summary>
    /// Uses stratified sampling to shuffle the indices for cross validation
    /// http://en.wikipedia.org/wiki/Stratified_sampling
    /// </summary>
    /// <typeparam name="TPredict"></typeparam>
    public sealed class StratifiedCrossValidation<TPrediction> : CrossValidation<TPrediction>
    {
        /// <summary>
        /// Cross validation for evaluating how learning algorithms generalise on new data
        /// </summary>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public StratifiedCrossValidation(int crossValidationFolds)
            : base(new StratifiedIndexSampler<double>(), crossValidationFolds)
        {
        }
    }
}
