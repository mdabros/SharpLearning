using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Samplers;
using System;
using System.Diagnostics;
using System.Linq;

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
            if (sampler == null) { throw new ArgumentNullException("shuffler"); }
            if (crossValidationFolds < 1) { throw new ArgumentException("CrossValidationFolds "); }

            m_indexedSampler = sampler;
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
            var rows = targets.Length;
            if (m_crossValidationFolds > rows) 
            {
                throw new ArgumentException("Too few observations: " + rows + 
                " for number of cross validation folds: " + m_crossValidationFolds); 
            }

            var holdOutSamples = new int[m_crossValidationFolds][];
            var samplesPrFold = rows / m_crossValidationFolds;
            var indices = Enumerable.Range(0, targets.Length).ToArray();

            for (int i = 0; i < m_crossValidationFolds; i++)
            {
                if(i == m_crossValidationFolds - 1)
                {
                    // last fold. Add remaining indices.  
                    holdOutSamples[i] = indices.ToArray();
                }
                else
                {
                    var holdoutSample = m_indexedSampler.Sample(targets, samplesPrFold, indices);
                    holdOutSamples[i] = holdoutSample;
                    indices = indices.Except(holdoutSample).ToArray();
                }
            }
         
            var crossValidatedPredictions = new TPrediction[rows];
            var allIndices = Enumerable.Range(0, targets.Length).ToArray();
            
            for (int i = 0; i < m_crossValidationFolds; i++)
            {
                var holdoutIndices = holdOutSamples[i];
                var trainingIndices = allIndices.Except(holdoutIndices).ToArray();
                var model = learner.Learn(observations, targets, trainingIndices);
                var predictions = new TPrediction[holdoutIndices.Length];

                for (int l = 0; l < predictions.Length; l++)
                {
                    predictions[l] = model.Predict(observations.GetRow(holdoutIndices[l]));  
                }

                for (int j = 0; j < holdoutIndices.Length; j++)
                {
                    crossValidatedPredictions[holdoutIndices[j]] = predictions[j];
                }
            }

            return crossValidatedPredictions;
        }
    }
}
