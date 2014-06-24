using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Shufflers;
using System;

namespace SharpLearning.CrossValidation
{
    /// <summary>
    /// Random cross validation uses a random shuffle of the observation indices to avoid any ordering issues.
    /// </summary>
    /// <typeparam name="TOut"></typeparam>
    public class RandomCrossValidation<TOut> : CrossValidation<TOut>
    {
        /// <summary>
        /// Cross validation for evaluating how learning algorithms perform on unseen observations
        /// </summary>
        /// <param name="modelCreator">The func should provide a learning algorithm 
        /// that returns a model predicting multiple observations</param>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public RandomCrossValidation(Func<F64Matrix, double[], int[], Func<F64Matrix, int[], TOut[]>> modelCreator,
                                        int crossValidationFolds)
            : base(modelCreator, new RandomCrossValidationShuffler(DateTime.Now.Millisecond), crossValidationFolds)
        {
        }

        /// <summary>
        /// Cross validation for evaluating how learning algorithms perform on unseen observations
        /// </summary>
        /// <param name="modelCreator">The func should provide a learning algorithm 
        /// that returns a model predicting multiple observations</param>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        /// <param name="seed"></param>
        public RandomCrossValidation(Func<F64Matrix, double[], int[], Func<F64Matrix, int[], TOut[]>> modelCreator,
                                        int crossValidationFolds, int seed)
            : base(modelCreator, new RandomCrossValidationShuffler(seed), crossValidationFolds)
        {
        }
    }
}
