using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation
{
    /// <summary>
    /// Evaluator delegate for cross validation
    /// </summary>
    /// <typeparam name="T">The prediction type</typeparam>
    /// <param name="observation">Single observation</param>
    /// <returns></returns>
    public delegate T CrossValidationEvaluator<T>(double[] observation);
    
 
    
    /// <summary>
    /// Learner to be used in cross validation
    /// </summary>
    /// <typeparam name="T">The target type</typeparam>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices">Indices to be used from observations</param>
    /// <returns></returns>
    public delegate CrossValidationEvaluator<TOut> CrossValidationLearner<TOut, TTarget>(F64Matrix observations, 
                                                                                                TTarget[] targets, int[] indices);
}