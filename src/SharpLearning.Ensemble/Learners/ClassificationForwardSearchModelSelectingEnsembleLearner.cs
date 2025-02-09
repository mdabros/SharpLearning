using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Ensemble.Learners;

/// <summary>
/// Classification model selecting EnsembleLearner. 
/// Trains several models and selects the best subset of models for the ensemble using greedy forward selection.
/// The selection of the best set of models is based on cross validation. 
/// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
/// </summary>
public sealed class ClassificationForwardSearchModelSelectingEnsembleLearner : ClassificationModelSelectingEnsembleLearner
{
    /// <summary>
    /// Classification model selecting EnsembleLearner. 
    /// Trains several models and selects the best subset of models for the ensemble using greedy forward selection.
    /// The selection of the best set of models is based on cross validation. 
    /// Default is 5-fold StratifiedCrossValidation and minimization of multi-class log loss and mean of probabilities is used to combine the models.
    /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    /// </summary>
    /// <param name="learners">Learners in the ensemble</param>
    /// <param name="numberOfModelsToSelect">Number of models to select</param>
    public ClassificationForwardSearchModelSelectingEnsembleLearner(
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
    public ClassificationForwardSearchModelSelectingEnsembleLearner(
        IIndexedLearner<ProbabilityPrediction>[] learners,
        int numberOfModelsToSelect,
        ICrossValidation<ProbabilityPrediction> crossValidation,
        IClassificationEnsembleStrategy ensembleStrategy,
        IMetric<double, ProbabilityPrediction> metric,
        int numberOfModelsFromStart = 1,
        bool selectWithReplacement = true)
        : base(learners, crossValidation, ensembleStrategy,
              new ForwardSearchClassificationEnsembleSelection(
                  metric, ensembleStrategy, numberOfModelsToSelect,
                  numberOfModelsFromStart, selectWithReplacement))
    {
    }
}
