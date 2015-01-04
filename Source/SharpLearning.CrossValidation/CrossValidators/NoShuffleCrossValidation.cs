using SharpLearning.CrossValidation.Shufflers;

namespace SharpLearning.CrossValidation.CrossValidators
{
    /// <summary>
    /// No shuffle cross validation does not shuffle the the observation indices and keeps the original order.
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public sealed class NoShuffleCrossValidation<TPrediction> : CrossValidation<TPrediction>
    {
        /// <summary>
        /// Cross validation for evaluating how learning algorithms generalise on new data
        /// </summary>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public NoShuffleCrossValidation(int crossValidationFolds)
            : base(new NoShuffleCrossValidationShuffler<double>(), crossValidationFolds)
        {
        }
    }
}
