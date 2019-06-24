using System;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.CrossValidators
{
    /// <summary>
    /// Random cross validation uses a random shuffle of the observation indices to avoid any ordering issues.
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public class RandomCrossValidation<TPrediction> : CrossValidation<TPrediction>
    {
        /// <summary>
        /// Cross validation for evaluating how learning algorithms generalize on new data
        /// </summary>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public RandomCrossValidation(int crossValidationFolds)
            : base(new RandomIndexSampler<double>(DateTime.Now.Millisecond), crossValidationFolds)
        {
        }

        /// <summary>
        /// Cross validation for evaluating how learning algorithms generalize on new data
        /// </summary>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        /// <param name="seed"></param>
        public RandomCrossValidation(int crossValidationFolds, int seed)
            : base(new RandomIndexSampler<double>(seed), crossValidationFolds)
        {
        }
    }
}
