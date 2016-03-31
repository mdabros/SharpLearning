using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Models;
using SharpLearning.Ensemble.Strategies;
using System;
using System.Linq;

namespace SharpLearning.Ensemble.Learners
{
    /// <summary>
    /// Classification model selecting EnsembleLearner.
    /// http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf
    /// </summary>
    public sealed class ClassificationModelSelectingEnsembleLearner : ILearner<ProbabilityPrediction>, IIndexedLearner<ProbabilityPrediction>,
        ILearner<double>, IIndexedLearner<double>
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
        public ClassificationModelSelectingEnsembleLearner(IIndexedLearner<ProbabilityPrediction>[] learners, ICrossValidation<ProbabilityPrediction> crossValidation,
            IClassificationEnsembleStrategy ensembleStrategy, IClassificationEnsembleSelection ensembleSelection)
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
        public ClassificationModelSelectingEnsembleLearner(IIndexedLearner<ProbabilityPrediction>[] learners, ICrossValidation<ProbabilityPrediction> crossValidation,
            Func<IClassificationEnsembleStrategy> ensembleStrategy, IClassificationEnsembleSelection ensembleSelection)
        {
            if (learners == null) { throw new ArgumentNullException("learners"); }
            if (crossValidation == null) { throw new ArgumentNullException("crossValidation"); }
            if (ensembleStrategy == null) { throw new ArgumentNullException("ensembleStrategy"); }
            if (ensembleSelection == null) { throw new ArgumentNullException("ensembleSelection"); }
            m_learners = learners;
            m_crossValidation = crossValidation;
            m_ensembleStrategy = ensembleStrategy;
            m_ensembleSelection = ensembleSelection;
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
        public ClassificationEnsembleModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            var metaObservations = LearnMetaFeatures(observations, targets, indices);

            var metaModelTargets = targets.GetIndices(indices);
            var ensembleModelIndices = m_ensembleSelection.Select(metaObservations, metaModelTargets);

            var ensembleModels = m_learners.GetIndices(ensembleModelIndices)
                .Select(learner => learner.Learn(observations, targets, indices)).ToArray();

            var numberOfClasses = targets.Distinct().Count();
            return new ClassificationEnsembleModel(ensembleModels, m_ensembleStrategy());
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
        IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
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
        IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
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
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        /// <summary>
        /// Learns a ClassificationEnsembleModel based on model selection.
        /// Trains several models and selects the best subset of models for the ensemble.
        /// The selection of the best set of models is based on cross validation.
        /// Trains several models and selects the best subset of models for the ensemble.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
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
        public ProbabilityPrediction[][] LearnMetaFeatures(F64Matrix observations, double[] targets, int[] indices)
        {
            var cvRows = indices.Length;
            var cvCols = m_learners.Length;

            var ensemblePredictions = new ProbabilityPrediction[cvCols][];

            for (int i = 0; i < m_learners.Length; i++)
            {
                var learner = m_learners[i];
                var learnerPredictions = new ProbabilityPrediction[cvRows];
                m_crossValidation.CrossValidate(learner, observations, targets, indices, learnerPredictions);
                ensemblePredictions[i] = learnerPredictions;
            }

            return ensemblePredictions;
        }
    }
}
