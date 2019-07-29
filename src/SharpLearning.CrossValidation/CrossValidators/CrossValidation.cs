using System;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.CrossValidators
{
    /// <summary>
    /// Cross validation for evaluating how learning algorithms perform on unseen observations
    /// </summary>
    /// <typeparam name="TPrediction"></typeparam>
    public class CrossValidation<TPrediction> : ICrossValidation<TPrediction>
    {
        readonly int m_crossValidationFolds;
        readonly IIndexSampler<double> m_indexedSampler;

        /// <summary>
        /// Cross validation for evaluating how learning algorithms perform on unseen observations
        /// </summary>
        /// <param name="sampler">Sampling strategy for the provided indices 
        /// before they are divided into the provided folds</param>
        /// <param name="crossValidationFolds">Number of folds that should be used for cross validation</param>
        public CrossValidation(IIndexSampler<double> sampler, int crossValidationFolds)
        {
            m_indexedSampler = sampler ?? throw new ArgumentNullException(nameof(sampler));
            if (crossValidationFolds < 1) { throw new ArgumentException("CrossValidationFolds "); }

            m_crossValidationFolds = crossValidationFolds;
        }

        /// <summary>
        /// Returns an array of cross validated predictions
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public TPrediction[] CrossValidate(IIndexedLearner<TPrediction> learner,
            F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            var cvPredictions = new TPrediction[indices.Length];

            CrossValidate(learner, observations, targets, indices, cvPredictions);

            return cvPredictions;
        }

        /// <summary>
        /// Cross validated predictions. 
        /// Only crossValidates within the provided indices.
        /// The predictions are returned in the predictions array.
        /// </summary>
        /// <param name="learner"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="crossValidationIndices"></param>
        /// <param name="crossValidatedPredictions"></param>
        public void CrossValidate(IIndexedLearner<TPrediction> learner,
            F64Matrix observations,
            double[] targets,
            int[] crossValidationIndices,
            TPrediction[] crossValidatedPredictions)
        {
            var rows = crossValidatedPredictions.Length;
            if (m_crossValidationFolds > rows)
            {
                throw new ArgumentException("Too few observations: " + rows +
                " for number of cross validation folds: " + m_crossValidationFolds);
            }

            var indices = crossValidationIndices.ToArray();

            // Map the provided crossValidationIndices to crossValidatedPredictions
            // Indices from crossValidationIndices can be larger than crossValidatedPredictions length
            // since crossValidatedPredictions might be a subset of the provided observations and targets
            var cvPredictionIndiceMap = Enumerable.Range(0, crossValidatedPredictions.Length)
                .ToDictionary(i => indices[i], i => i);

            var crossValidationIndexSets = CrossValidationUtilities.GetKFoldCrossValidationIndexSets(
                m_indexedSampler, m_crossValidationFolds, targets, indices);

            var observation = new double[observations.ColumnCount];
            foreach (var (trainingIndices, validationIndices) in crossValidationIndexSets)
            {
                var model = learner.Learn(observations, targets, trainingIndices);
                var predictions = new TPrediction[validationIndices.Length];

                for (int l = 0; l < predictions.Length; l++)
                {
                    observations.Row(validationIndices[l], observation);
                    predictions[l] = model.Predict(observation);
                }

                for (int j = 0; j < validationIndices.Length; j++)
                {
                    crossValidatedPredictions[cvPredictionIndiceMap[validationIndices[j]]] = predictions[j];
                }

                ModelDisposer.DisposeIfDisposable(model);
            }
        }
    }
}
