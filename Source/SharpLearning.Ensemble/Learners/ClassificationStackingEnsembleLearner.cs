using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.Ensemble.Models;
using System;
using System.Linq;

namespace SharpLearning.Ensemble.Learners
{
    /// <summary>
    /// Stacking Classification Ensemble Learner.
    /// http://mlwave.com/kaggle-ensembling-guide/
    /// </summary>
    public class ClassificationStackingEnsembleLearner : ILearner<ProbabilityPrediction>, IIndexedLearner<ProbabilityPrediction>, 
        ILearner<double>, IIndexedLearner<double>
    {
        readonly IIndexedLearner<ProbabilityPrediction>[] m_learners;
        readonly ICrossValidation<ProbabilityPrediction> m_crossValidation;
        readonly Func<F64Matrix, double[], IPredictorModel<ProbabilityPrediction>> m_metaLearner;
        readonly bool m_includeOriginalFeaturesForMetaLearner;

        /// <summary>
        /// Stacking Classification Ensemble Learner. 
        /// Combines several models into a single ensemble model using a top or meta level model to combine the models.
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="crossValidation">Cross validation method</param>
        /// <param name="metaLearner">Meta learner or top level model for combining the ensemble models</param>
        /// <param name="includeOriginalFeaturesForMetaLearner">True; the meta learner also recieves the original features. 
        /// False; the meta learner only recieves the output of the ensemble models as features</param>
        public ClassificationStackingEnsembleLearner(IIndexedLearner<ProbabilityPrediction>[] learners, ICrossValidation<ProbabilityPrediction> crossValidation,
            ILearner<ProbabilityPrediction> metaLearner, bool includeOriginalFeaturesForMetaLearner)
            : this(learners, crossValidation, (obs, targets) => metaLearner.Learn(obs, targets), includeOriginalFeaturesForMetaLearner)
        {
        }

        /// <summary>
        /// Stacking Classification Ensemble Learner. 
        /// Combines several models into a single ensemble model using a top or meta level model to combine the models.
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="crossValidation">Cross validation method</param>
        /// <param name="metaLearner">Meta learner or top level model for combining the ensemble models</param>
        /// <param name="includeOriginalFeaturesForMetaLearner">True; the meta learner also recieves the original features. 
        /// False; the meta learner only recieves the output of the ensemble models as features</param>
        public ClassificationStackingEnsembleLearner(IIndexedLearner<ProbabilityPrediction>[] learners, ICrossValidation<ProbabilityPrediction> crossValidation,
            Func<F64Matrix, double[], IPredictorModel<ProbabilityPrediction>> metaLearner, bool includeOriginalFeaturesForMetaLearner)
        {
            if (learners == null) { throw new ArgumentException("learners"); }
            if (crossValidation == null) { throw new ArgumentException("crossValidation"); }
            if (metaLearner == null) { throw new ArgumentException("metaLearner"); }
            m_learners = learners;
            m_crossValidation = crossValidation;
            m_metaLearner = metaLearner;
            m_includeOriginalFeaturesForMetaLearner = includeOriginalFeaturesForMetaLearner;
        }

        /// <summary>
        /// Learns a stacking classification ensemble
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationStackingEnsembleModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// /// Learns a stacking classification ensemble on the provided indices
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ClassificationStackingEnsembleModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            var metaObservations = LearnMetaFeatures(observations, targets, indices);

            var metaModelTargets = targets.GetIndices(indices);
            var metaModel = m_metaLearner(metaObservations, metaModelTargets);
            var ensembleModels = m_learners.Select(learner => learner.Learn(observations, targets, indices)).ToArray();

            var numberOfClasses = targets.Distinct().Count();
            return new ClassificationStackingEnsembleModel(ensembleModels, metaModel, m_includeOriginalFeaturesForMetaLearner, numberOfClasses);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        /// <summary>
        /// 
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
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        /// <summary>
        /// 
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
        public F64Matrix LearnMetaFeatures(F64Matrix observations, double[] targets, int[] indices)
        {
            var numberOfClasses = targets.Distinct().Count();

            var cvRows = indices.Length;
            var ensembleFeatures = m_learners.Length * numberOfClasses;
            var cvCols = ensembleFeatures;

            if (m_includeOriginalFeaturesForMetaLearner)
            {
                cvCols = cvCols + observations.GetNumberOfColumns();
            }

            var cvPredictions = new F64Matrix(cvRows, cvCols);
            var modelPredictions = new ProbabilityPrediction[cvRows];
            for (int i = 0; i < m_learners.Length; i++)
            {
                var learner = m_learners[i];
                m_crossValidation.CrossValidate(learner, observations, targets, indices, modelPredictions);
                for (int j = 0; j < modelPredictions.Length; j++)
                {
                    var probabilities = modelPredictions[j].Probabilities.Values.ToArray();
                    for (int k = 0; k < probabilities.Length; k++)
                    {
                        cvPredictions[j, i * numberOfClasses + k] = probabilities[k];
                    }
                }
            }

            if (m_includeOriginalFeaturesForMetaLearner)
            {
                for (int i = 0; i < cvRows; i++)
                {
                    for (int j = 0; j < observations.GetNumberOfColumns(); j++)
                    {
                        var value = cvPredictions[i, j + ensembleFeatures];
                        cvPredictions[i, j + ensembleFeatures] = observations[indices[i], j];
                    }
                }
            }

            return cvPredictions;
        }
    }
}
