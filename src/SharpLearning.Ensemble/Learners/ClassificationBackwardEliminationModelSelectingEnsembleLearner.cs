using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Learners
{
    /// <summary>
    /// Classification model selecting EnsembleLearner. 
    /// Trains several models and selects the best subset of models for the ensemble using greedy backward elimination.
    /// The selection of the best set of models is based on cross validation. 
    /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    /// </summary>
    public sealed class ClassificationBackwardEliminationModelSelectingEnsembleLearner : ClassificationModelSelectingEnsembleLearner
    {
        /// <summary>
        /// Classification model selecting EnsembleLearner. 
        /// Trains several models and selects the best subset of models for the ensemble using greedy backward elimination.
        /// The selection of the best set of models is based on cross validation. 
        /// Default is 5-fold StratifiedCrossValidation and minimization of multi-class log loss and mean of probabilities is used to combine the models.
        /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        public ClassificationBackwardEliminationModelSelectingEnsembleLearner(
            IIndexedLearner<ProbabilityPrediction>[] learners, 
            int numberOfModelsToSelect)
            : this(learners, numberOfModelsToSelect, 
                new StratifiedCrossValidation<ProbabilityPrediction>(5, 42), 
                new MeanProbabilityClassificationEnsembleStrategy(),
                new LogLossClassificationProbabilityMetric())
        {
        }

        /// <summary>
        /// Classification model selecting EnsembleLearner. 
        /// Trains several models and selects the best subset of models for the ensemble using greedy backward elimination.
        /// The selection of the best set of models is based on cross validation. 
        /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        /// <param name="crossValidation">Cross validation method</param>
        /// <param name="ensembleStrategy">Strategy for ensembling models</param>
        /// <param name="metric">Metric to minimize</param>
        public ClassificationBackwardEliminationModelSelectingEnsembleLearner(
            IIndexedLearner<ProbabilityPrediction>[] learners, 
            int numberOfModelsToSelect,
            ICrossValidation<ProbabilityPrediction> crossValidation, 
            IClassificationEnsembleStrategy ensembleStrategy, 
            IMetric<double, ProbabilityPrediction> metric)
            : base(learners, crossValidation, ensembleStrategy, 
                  new BackwardEliminationClassificationEnsembleSelection(
                      metric, ensembleStrategy, numberOfModelsToSelect))
        {
        }
    }
}
