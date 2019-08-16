using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.CrossValidators
{
    /// <summary>
    /// No shuffle cross validation does not shuffle the observation indices and keeps the original order.
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public sealed class NoShuffleCrossValidation<TPrediction> : CrossValidation<TPrediction>
    {
        /// <summary>
        /// Cross validation for evaluating how learning algorithms generalize on new data
        /// </summary>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public NoShuffleCrossValidation(int crossValidationFolds)
            : base(new NoShuffleIndexSampler<double>(), crossValidationFolds)
        {
        }
    }
}
