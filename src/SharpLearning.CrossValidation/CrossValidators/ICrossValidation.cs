using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation.CrossValidators
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="TPredicion"></typeparam>
    /// <param name="observation"></param>
    /// <returns></returns>
    public delegate TPredicion ModelPredictor<TPredicion>(double[] observation);

    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="TPredicion"></typeparam>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    public delegate ModelPredictor<TPredicion> ModelLearner<TPredicion>(F64Matrix observations, 
        double[] targets, int[] indices);

    /// <summary>
    /// Cross validation interface for evaluating how learning algorithms perform on unseen observations
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public interface ICrossValidation<TPrediction>
    {
        /// <summary>
        /// Returns an array of cross validated predictions
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        TPrediction[] CrossValidate(IIndexedLearner<TPrediction> learner,
            F64Matrix observations, double[] targets);

        /// <summary>
        /// Cross validated predictions. 
        /// Only crossValidates within the provided indices.
        /// The predictions are returned in the predictions array.
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="crossValidationIndices"></param>
        /// <param name="crossValidatedPredictions"></param>
        void CrossValidate(IIndexedLearner<TPrediction> learner,
            F64Matrix observations,
            double[] targets,
            int[] crossValidationIndices,
            TPrediction[] crossValidatedPredictions);

        /// <summary>
        /// Returns an array of cross validated predictions
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        TPrediction[] CrossValidate(ModelLearner<TPrediction> learner,
            F64Matrix observations, double[] targets);

        /// <summary>
        /// Cross validated predictions. 
        /// Only crossValidates within the provided indices.
        /// The predictions are returned in the predictions array.
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="crossValidationIndices"></param>
        /// <param name="crossValidatedPredictions"></param>
        void CrossValidate(ModelLearner<TPrediction> learner,
            F64Matrix observations, 
            double[] targets, 
            int[] crossValidationIndices, 
            TPrediction[] crossValidatedPredictions);
    }
}
