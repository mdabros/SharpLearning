using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.Shufflers;
using SharpLearning.CrossValidation.TrainingValidationSplitters;

namespace SharpLearning.CrossValidation.BiasVarianceAnalysis
{
    /// <summary>
    /// Bias variance analysis calculator for constructing learning curves.
    /// Learning curves can be used to determine if a model has high bias or high variance.
    /// 
    /// The order of the data is stratified to have similar destributions in training and validation set.
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
    public sealed class StratifiedBiasVarianceLearningCurvesCalculator<TPrediction> : 
        BiasVarianceLearningCurvesCalculator<TPrediction>
    {
        /// <summary>
        /// Bias variance analysis calculator for constructing learning curves.
        /// Learning curves can be used to determine if a model has high bias or high variance.
        /// 
        /// The order of the data is stratified to have similar destributions in training and validation set.
        /// </summary>
        /// <param name="metric">The error metric used</param>
        /// <param name="samplePercentages">A list of sample percentages determining the 
        /// training data used in each point of the learning curve</param>
        /// <param name="numberOfShufflesPrSample">Number of shuffles done at each sampling point</param>
        public StratifiedBiasVarianceLearningCurvesCalculator(IMetric<double, TPrediction> metric, double[] samplePercentages,
            double trainingPercentage, int numberOfShufflesPrSample = 5)
            : base(new StratifiedTrainingValidationIndexSplitter<double>(trainingPercentage),
                   new StratifyCrossValidationShuffler<double>(), metric, samplePercentages, numberOfShufflesPrSample)
        {
        }

        /// <summary>
        /// Bias variance analysis calculator for constructing learning curves.
        /// Learning curves can be used to determine if a model has high bias or high variance.
        /// 
        /// The order of the data is stratified to have similar destributions in training and validation set.
        /// </summary>
        /// <param name="metric">The error metric used</param>
        /// <param name="samplePercentages">A list of sample percentages determining the 
        /// <param name="trainingPercentage"></param>
        /// <param name="seed"></param>
        /// <param name="numberOfShufflesPrSample">Number of shuffles done at each sampling point</param>
        public StratifiedBiasVarianceLearningCurvesCalculator(IMetric<double, TPrediction> metric, double[] samplePercentages,
            double trainingPercentage, int seed, int numberOfShufflesPrSample = 5)
            : base(new StratifiedTrainingValidationIndexSplitter<double>(trainingPercentage),
                   new StratifyCrossValidationShuffler<double>(seed), metric, samplePercentages, numberOfShufflesPrSample)
        {
        }
    }
}
