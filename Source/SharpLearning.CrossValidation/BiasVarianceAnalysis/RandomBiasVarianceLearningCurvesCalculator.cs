using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.Shufflers;
using SharpLearning.CrossValidation.TrainingValidationSplitters;

namespace SharpLearning.CrossValidation.BiasVarianceAnalysis
{
    /// <summary>
    /// Bias variance analysis calculator for constructing learning curves.
    /// Learning curves can be used to determine if a model has high bias or high variance.
    /// 
    /// The order of the data is randomized.
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
    public sealed class RandomBiasVarianceLearningCurvesCalculator<TPrediction> : 
        BiasVarianceLearningCurvesCalculator<TPrediction>
    {
        /// <summary>
        /// Bias variance analysis calculator for constructing learning curves.
        /// Learning curves can be used to determine if a model has high bias or high variance.
        /// 
        /// The order of the data is randomized.
        /// </summary>
        /// <param name="metric">The error metric used</param>
        /// <param name="samplePercentages">A list of sample percentages determining the 
        /// training data used in each point of the learning curve</param>
        public RandomBiasVarianceLearningCurvesCalculator(IMetric<double, TPrediction> metric, double[] samplePercentages, 
            double trainingPercentage)
            : base(new RandomTrainingValidationIndexSplitter<double>(trainingPercentage), 
                   new RandomCrossValidationShuffler<double>(),  metric, samplePercentages)
        {
        }

        /// <summary>
        /// Bias variance analysis calculator for constructing learning curves.
        /// Learning curves can be used to determine if a model has high bias or high variance.
        /// 
        /// The order of the data is randomized.
        /// </summary>
        /// <param name="metric">The error metric used</param>
        /// <param name="samplePercentages">A list of sample percentages determining the 
        /// <param name="trainingPercentage"></param>
        /// <param name="seed"></param>
        public RandomBiasVarianceLearningCurvesCalculator(IMetric<double, TPrediction> metric, double[] samplePercentages, 
            double trainingPercentage, int seed)
            : base(new RandomTrainingValidationIndexSplitter<double>(trainingPercentage, seed), 
            new RandomCrossValidationShuffler<double>(seed), metric, samplePercentages)
        {
        }
    }
}
