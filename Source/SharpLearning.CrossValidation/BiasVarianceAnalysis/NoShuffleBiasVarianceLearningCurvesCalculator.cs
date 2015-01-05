using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.TrainingValidationSplitters;

namespace SharpLearning.CrossValidation.BiasVarianceAnalysis
{
    /// <summary>
    /// Bias variance analysis calculator for constructing learning curves.
    /// Learning curves can be used to determine if a model has high bias or high variance.
    /// 
    /// The order of the data is kept when splitting the data.
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
    public sealed class NoShuffleBiasVarianceLearningCurvesCalculator<TPrediction> : 
        BiasVarianceLearningCurvesCalculator<TPrediction>
    {
        /// <summary>
        /// Bias variance analysis calculator for constructing learning curves.
        /// Learning curves can be used to determine if a model has high bias or high variance.
        /// 
        /// The order of the data is kept when splitting the data.
        /// </summary>
        /// <param name="metric">The error metric used</param>
        /// <param name="samplePercentages">A list of sample percentages determining the 
        /// training data used in each point of the learning curve</param>
        public NoShuffleBiasVarianceLearningCurvesCalculator(IMetric<double, TPrediction> metric, double[] samplePercentages, double trainingPercentage)
            : base(new NoShuffleTrainingValidationIndexSplitter<double>(trainingPercentage), metric, samplePercentages)
        {
        }
    }
}
