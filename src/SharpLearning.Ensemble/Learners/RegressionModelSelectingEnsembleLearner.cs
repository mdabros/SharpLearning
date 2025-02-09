using System;
using System.Diagnostics;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Models;
using SharpLearning.Ensemble.Strategies;

namespace SharpLearning.Ensemble.Learners;

/// <summary>
/// Regression model selecting EnsembleLearner.
/// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
/// </summary>
public class RegressionModelSelectingEnsembleLearner : ILearner<double>, IIndexedLearner<double>
{
    readonly IIndexedLearner<double>[] m_learners;
    readonly ICrossValidation<double> m_crossValidation;
    readonly Func<IRegressionEnsembleStrategy> m_ensembleStrategy;
    readonly IRegressionEnsembleSelection m_ensembleSelection;

    /// <summary>
    /// Regression model selecting EnsembleLearner. 
    /// Trains several models and selects the best subset of models for the ensemble.
    /// The selection of the best set of models is based on cross validation.
    /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    /// </summary>
    /// <param name="learners">Learners in the ensemble</param>
    /// <param name="crossValidation">Cross validation method</param>
    /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
    /// <param name="ensembleSelection">Ensemble selection method used to find the beset subset of models</param>
    public RegressionModelSelectingEnsembleLearner(
        IIndexedLearner<double>[] learners,
        ICrossValidation<double> crossValidation,
        IRegressionEnsembleStrategy ensembleStrategy,
        IRegressionEnsembleSelection ensembleSelection)
        : this(learners, crossValidation, () => ensembleStrategy, ensembleSelection)
    {
    }

    /// <summary>
    /// Regression model selecting EnsembleLearner. 
    /// Trains several models and selects the best subset of models for the ensemble.
    /// The selection of the best set of models is based on cross validation.
    /// Trains several models and selects the best subset of models for the ensemble.
    /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    /// </summary>
    /// <param name="learners">Learners in the ensemble</param>
    /// <param name="crossValidation">Cross validation method</param>
    /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
    /// <param name="ensembleSelection">Ensemble selection method used to find the beset subset of models</param>
    public RegressionModelSelectingEnsembleLearner(
        IIndexedLearner<double>[] learners,
        ICrossValidation<double> crossValidation,
        Func<IRegressionEnsembleStrategy> ensembleStrategy,
        IRegressionEnsembleSelection ensembleSelection)
    {
        m_learners = learners ?? throw new ArgumentNullException(nameof(learners));
        m_crossValidation = crossValidation ?? throw new ArgumentNullException(nameof(crossValidation));
        m_ensembleStrategy = ensembleStrategy ?? throw new ArgumentNullException(nameof(ensembleStrategy));
        m_ensembleSelection = ensembleSelection ?? throw new ArgumentNullException(nameof(ensembleSelection));
    }

    /// <summary>
    /// Learns a RegressionEnsembleModel based on model selection.
    /// Trains several models and selects the best subset of models for the ensemble.
    /// The selection of the best set of models is based on cross validation.
    /// Trains several models and selects the best subset of models for the ensemble.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    public RegressionEnsembleModel Learn(F64Matrix observations, double[] targets)
    {
        var indices = Enumerable.Range(0, targets.Length).ToArray();
        return Learn(observations, targets, indices);
    }

    /// <summary>
    /// Learns a stacking classification ensemble on the provided indices
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    public RegressionEnsembleModel Learn(F64Matrix observations, double[] targets,
        int[] indices)
    {
        Checks.VerifyObservationsAndTargets(observations, targets);
        Checks.VerifyIndices(indices, observations, targets);

        var metaObservations = LearnMetaFeatures(observations, targets, indices);

        var metaModelTargets = targets.GetIndices(indices);
        var ensembleModelIndices = m_ensembleSelection.Select(metaObservations,
            metaModelTargets);

        var ensembleModels = m_learners.GetIndices(ensembleModelIndices)
            .Select(learner => learner.Learn(observations, targets, indices)).ToArray();

        return new RegressionEnsembleModel(ensembleModels, m_ensembleStrategy());
    }

    /// <summary>
    /// Learns and extracts the meta features learned by the ensemble models
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    public F64Matrix LearnMetaFeatures(F64Matrix observations, double[] targets)
    {
        var indices = Enumerable.Range(0, targets.Length).ToArray();
        return LearnMetaFeatures(observations, targets, indices);
    }

    /// <summary>
    /// Learns and extracts the meta features learned by the ensemble models on the provided indices
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    public F64Matrix LearnMetaFeatures(F64Matrix observations, double[] targets,
        int[] indices)
    {
        var cvRows = indices.Length;
        var cvCols = m_learners.Length;

        var cvPredictions = new F64Matrix(cvRows, cvCols);
        var modelPredictions = new double[cvRows];
        for (int i = 0; i < m_learners.Length; i++)
        {
            Trace.WriteLine("Training model: " + (i + 1));
            var learner = m_learners[i];
            m_crossValidation.CrossValidate(learner, observations, targets,
                indices, modelPredictions);

            for (int j = 0; j < modelPredictions.Length; j++)
            {
                cvPredictions[j, i] = modelPredictions[j];
            }
        }

        return cvPredictions;
    }

    /// <summary>
    /// Based on the provided metaObservations selects the best combination of learners to include in the ensemble.
    /// Following the selected learners are trained.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="metaObservations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    public RegressionEnsembleModel SelectModels(
        F64Matrix observations,
        F64Matrix metaObservations,
        double[] targets)
    {
        var indices = Enumerable.Range(0, targets.Length).ToArray();
        var ensembleModelIndices = m_ensembleSelection.Select(metaObservations, targets);

        var ensembleModels = m_learners.GetIndices(ensembleModelIndices)
            .Select(learner => learner.Learn(observations, targets, indices)).ToArray();

        return new RegressionEnsembleModel(ensembleModels, m_ensembleStrategy());
    }

    /// <summary>
    /// Private explicit interface implementation for indexed learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    IPredictorModel<double> IIndexedLearner<double>.Learn(
        F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

    /// <summary>
    /// Private explicit interface implementation for learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    IPredictorModel<double> ILearner<double>.Learn(
        F64Matrix observations, double[] targets) => Learn(observations, targets);
}