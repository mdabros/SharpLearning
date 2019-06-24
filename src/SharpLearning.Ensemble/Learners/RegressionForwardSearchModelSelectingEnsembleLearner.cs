using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Learners
{
    /// <summary>
    /// Regression model selecting EnsembleLearner. 
    /// Trains several models and selects the best subset of models for the ensemble using greedy forward selection.
    /// The selection of the best set of models is based on cross validation.
    /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    /// </summary>
    public sealed class RegressionForwardSearchModelSelectingEnsembleLearner : RegressionModelSelectingEnsembleLearner
    {
        /// <summary>
        /// Regression model selecting EnsembleLearner. 
        /// Trains several models and selects the best subset of models for the ensemble using greedy forward selection.
        /// The selection of the best set of models is based on cross validation. 
        /// Default is 5-fold RandomCrossValidation and minimization of mean square error and mean is used to combine the models.
        /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        public RegressionForwardSearchModelSelectingEnsembleLearner(
            IIndexedLearner<double>[] learners, 
            int numberOfModelsToSelect)
            : this(learners, numberOfModelsToSelect, 
                new RandomCrossValidation<double>(5, 42), 
                new MeanRegressionEnsembleStrategy(),
                new MeanSquaredErrorRegressionMetric())
        {
        }

        /// <summary>
        /// Regression model selecting EnsembleLearner. 
        /// Trains several models and selects the best subset of models for the ensemble using greedy forward selection.
        /// The selection of the best set of models is based on cross validation. 
        /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        /// <param name="crossValidation">Cross validation method</param>
        /// <param name="ensembleStrategy">Strategy for ensembling models</param>
        /// <param name="metric">Metric to minimize</param>
        /// <param name="numberOfModelsFromStart">Number of models from start of the search. 
        /// The top n models will be selected based in their solo performance</param>
        /// <param name="selectWithReplacement">If true the same model can be selected multiple times.
        /// This will correspond to weighting the models. If false each model can only be selected once. Default is true</param>
        public RegressionForwardSearchModelSelectingEnsembleLearner(
            IIndexedLearner<double>[] learners, 
            int numberOfModelsToSelect,
            ICrossValidation<double> crossValidation, 
            IRegressionEnsembleStrategy ensembleStrategy, 
            IMetric<double, double> metric, 
            int numberOfModelsFromStart = 1, 
            bool selectWithReplacement = true)
            : base(learners, crossValidation, ensembleStrategy, 
                new ForwardSearchRegressionEnsembleSelection(metric, ensembleStrategy, numberOfModelsToSelect, 
                    numberOfModelsFromStart, selectWithReplacement))
        {
        }
    }
}
