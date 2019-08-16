using System.Collections.Generic;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.CrossValidation.LearningCurves
{
    /// <summary>
    /// Bias variance analysis calculator for constructing learning curves.
    /// Learning curves can be used to determine if a model has high bias or high variance.
    /// 
    /// Solutions for model with high bias:
    ///  - Add more features.
    ///  - Use a more sophisticated model
    ///  - Decrease regularization.
    /// Solutions for model with high variance
    ///  - Use fewer features.
    ///  - Use more training samples.
    ///  - Increase Regularization.
    /// </summary>
    public interface ILearningCurvesCalculator<TPrediction>
    {
        /// <summary>
        /// Returns a list of BiasVarianceLearningCurvePoints for constructing learning curves.
        /// The points contain sample size, training score and validation score. 
        /// </summary>
        /// <param name="learnerFactory"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        List<LearningCurvePoint> Calculate(IIndexedLearner<TPrediction> learnerFactory,
            F64Matrix observations, double[] targets);
    }
}
