﻿using System;
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

namespace SharpLearning.Ensemble.Learners
{
    /// <summary>
    /// Classification model selecting EnsembleLearner.
    /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    /// </summary>
    public class ClassificationModelSelectingEnsembleLearner 
        : ILearner<ProbabilityPrediction>
        , IIndexedLearner<ProbabilityPrediction>
        , ILearner<double>
        , IIndexedLearner<double>
    {
        readonly IIndexedLearner<ProbabilityPrediction>[] m_learners;
        readonly ICrossValidation<ProbabilityPrediction> m_crossValidation;
        readonly Func<IClassificationEnsembleStrategy> m_ensembleStrategy;
        readonly IClassificationEnsembleSelection m_ensembleSelection;

        /// <summary>
        /// Classification model selecting EnsembleLearner. 
        /// Trains several models and selects the best subset of models for the ensemble.
        /// The selection of the best set of models is based on cross validation.
        /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="crossValidation">Cross validation method</param>
        /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
        /// <param name="ensembleSelection">Ensemble selection method used to find the beset subset of models</param>
        public ClassificationModelSelectingEnsembleLearner(
            IIndexedLearner<ProbabilityPrediction>[] learners, 
            ICrossValidation<ProbabilityPrediction> crossValidation,
            IClassificationEnsembleStrategy ensembleStrategy, 
            IClassificationEnsembleSelection ensembleSelection)
            : this(learners, crossValidation, () => ensembleStrategy, ensembleSelection)
        {
        }

        /// <summary>
        /// Classification model selecting EnsembleLearner. 
        /// Trains several models and selects the best subset of models for the ensemble.
        /// The selection of the best set of models is based on cross validation.
        /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="crossValidation">Cross validation method</param>
        /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
        /// <param name="ensembleSelection">Ensemble selection method used to find the beset subset of models</param>
        public ClassificationModelSelectingEnsembleLearner(
            IIndexedLearner<ProbabilityPrediction>[] learners, 
            ICrossValidation<ProbabilityPrediction> crossValidation,
            Func<IClassificationEnsembleStrategy> ensembleStrategy, 
            IClassificationEnsembleSelection ensembleSelection)
        {
            m_learners = learners ?? throw new ArgumentNullException(nameof(learners));
            m_crossValidation = crossValidation ?? throw new ArgumentNullException(nameof(crossValidation));
            m_ensembleStrategy = ensembleStrategy ?? throw new ArgumentNullException(nameof(ensembleStrategy));
            m_ensembleSelection = ensembleSelection ?? throw new ArgumentNullException(nameof(ensembleSelection));
        }
        
        /// <summary>
        /// Learns a ClassificationEnsembleModel based on model selection.
        /// Trains several models and selects the best subset of models for the ensemble.
        /// The selection of the best set of models is based on cross validation.
        /// Trains several models and selects the best subset of models for the ensemble.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationEnsembleModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a ClassificationEnsembleModel based on model selection using the provided indices.
        /// Trains several models and selects the best subset of models for the ensemble.
        /// The selection of the best set of models is based on cross validation.
        /// Trains several models and selects the best subset of models for the ensemble.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ClassificationEnsembleModel Learn(F64Matrix observations, double[] targets, 
            int[] indices)
        {
            Checks.VerifyObservationsAndTargets(observations, targets);
            Checks.VerifyIndices(indices, observations, targets);

            var metaObservations = LearnMetaFeatures(observations, targets, indices);
            var metaModelTargets = targets.GetIndices(indices);

            var ensembleModelIndices = m_ensembleSelection.Select(metaObservations, metaModelTargets);

            var ensembleModels = m_learners.GetIndices(ensembleModelIndices)
                .Select(learner => learner.Learn(observations, targets, indices)).ToArray();

            var numberOfClasses = targets.Distinct().Count();
            return new ClassificationEnsembleModel(ensembleModels, m_ensembleStrategy());
        }

        /// <summary>
        /// Learns and extracts the meta features learned by the ensemble models
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ProbabilityPrediction[][] LearnMetaFeatures(F64Matrix observations, double[] targets)
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
        public ProbabilityPrediction[][] LearnMetaFeatures(F64Matrix observations, double[] targets, 
            int[] indices)
        {
            var cvRows = indices.Length;
            var cvCols = m_learners.Length;

            var ensemblePredictions = new ProbabilityPrediction[cvCols][];

            for (int i = 0; i < m_learners.Length; i++)
            {
                Trace.WriteLine("Training model: " + (i + 1));
                var learner = m_learners[i];
                var learnerPredictions = new ProbabilityPrediction[cvRows];
                m_crossValidation.CrossValidate(learner, observations, targets, indices, learnerPredictions);
                ensemblePredictions[i] = learnerPredictions;
            }

            return ensemblePredictions;
        }

        /// <summary>
        /// Based on the provided metaObservations selects the best combination of learners to include in the ensemble.
        /// Following the selected learners are trained.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="metaObservations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationEnsembleModel SelectModels(
            F64Matrix observations,
            ProbabilityPrediction[][] metaObservations, 
            double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            var ensembleModelIndices = m_ensembleSelection.Select(metaObservations, targets);

            var ensembleModels = m_learners.GetIndices(ensembleModelIndices)
                .Select(learner => learner.Learn(observations, targets, indices)).ToArray();

            var numberOfClasses = targets.Distinct().Count();
            return new ClassificationEnsembleModel(ensembleModels, m_ensembleStrategy());
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
        /// Private explicit interface implementation for indexed probability learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(
            F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

        /// <summary>
        /// Private explicit interface implementation.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(
            F64Matrix observations, double[] targets) => Learn(observations, targets);

        /// <summary>
        /// Private explicit interface implementation for probability learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(
            F64Matrix observations, double[] targets) => Learn(observations, targets);
    }
}
