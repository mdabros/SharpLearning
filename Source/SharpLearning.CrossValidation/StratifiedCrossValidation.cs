using SharpLearning.CrossValidation.Shufflers;
using System;

namespace SharpLearning.CrossValidation
{
    /// <summary>
    /// Uses stratified sampling to shuffle the indices for cross validation
    /// http://en.wikipedia.org/wiki/Stratified_sampling
    /// </summary>
    /// <typeparam name="TOut"></typeparam>
    public sealed class StratifiedCrossValidation<TOut, TTarget> : CrossValidation<TOut, TTarget>
    {
        /// <summary>
        /// Cross validation for evaluating how learning algorithms generalise on new data
        /// </summary>
        /// <param name="modelLearner">The func should provide a learning algorithm 
        /// that returns a model predicting multiple observations</param>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public StratifiedCrossValidation(CrossValidationLearner<TOut, TTarget> modelLearner, int crossValidationFolds)
            : base(modelLearner, new StratifyCrossValidationShuffler<TTarget>(DateTime.Now.Millisecond), crossValidationFolds)
        {
        }

        /// <summary>
        /// Cross validation for evaluating how learning algorithms generalise on new data
        /// </summary>
        /// <param name="modelLearner">The func should provide a learning algorithm 
        /// that returns a model predicting multiple observations</param>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        /// <param name="seed"></param>
        public StratifiedCrossValidation(CrossValidationLearner<TOut, TTarget> modelLearner, int crossValidationFolds, int seed)
            : base(modelLearner, new StratifyCrossValidationShuffler<TTarget>(seed), crossValidationFolds)
        {
        }
    }
}
