using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Ensemble.Strategies;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Ensemble.EnsembleSelectors
{
    /// <summary>
    /// Greedy backwards elimination of ensemble models.
    /// </summary>
    public sealed class BackwardEliminationClassificationEnsembleSelection : IClassificationEnsembleSelection
    {
        readonly IMetric<double, ProbabilityPrediction> m_metric;
        readonly IClassificationEnsembleStrategy m_ensembleStrategy;
        readonly int m_numberOfModelsToSelect;
        List<int> m_remainingModelIndices;
        List<int> m_bestModelIndices;

        /// <summary>
        /// Greedy backwards elimination of ensemble models.
        /// </summary>
        /// <param name="metric">Metric to minimize</param>
        /// <param name="ensembleStrategy">Strategy for ensembling models</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        public BackwardEliminationClassificationEnsembleSelection(IMetric<double, ProbabilityPrediction> metric, IClassificationEnsembleStrategy ensembleStrategy,
            int numberOfModelsToSelect)
        {
            if (metric == null) { throw new ArgumentNullException("metric"); }
            if (ensembleStrategy == null) { throw new ArgumentNullException("ensembleStrategy"); }
            if (numberOfModelsToSelect < 1) { throw new ArgumentException("numberOfModelsToSelect must be at least 1"); }
            m_metric = metric;
            m_ensembleStrategy = ensembleStrategy;
            m_numberOfModelsToSelect = numberOfModelsToSelect;
        }

        /// <summary>
        /// Greedy forward selection of ensemble models.
        /// </summary>
        /// <param name="crossValidatedModelPredictions">cross validated predictions from multiple models. 
        /// Each row in the matrix corresponds to predictions from a separate model</param>
        /// <param name="targets">Corresponding targets</param>
        /// <returns>The indices of the selected model</returns>
        public int[] Select(ProbabilityPrediction[][] crossValidatedModelPredictions, double[] targets)
        {
            if(crossValidatedModelPredictions.Length < m_numberOfModelsToSelect)
            {
                throw new ArgumentException("Availible models: " + crossValidatedModelPredictions.Length +
                    " is smaller than number of models to select: " + m_numberOfModelsToSelect);
            }

            m_remainingModelIndices = Enumerable.Range(0, crossValidatedModelPredictions.Length).ToList();

            var currentError = double.MaxValue;
            var modelsToRemove = m_remainingModelIndices.Count - 1;

            for (int i = 0; i < modelsToRemove; i++)
            {
                var error = SelectNextModelToRemove(crossValidatedModelPredictions, targets, currentError);
                Trace.WriteLine(error);
                if (error < currentError && m_remainingModelIndices.Count <= m_numberOfModelsToSelect)
                {
                    currentError = error;
                    m_bestModelIndices = m_remainingModelIndices.ToList();
                    Trace.WriteLine("Updated: " + error);
                }
            }

            Trace.WriteLine(string.Join(", ", m_bestModelIndices.ToArray()));

            return m_bestModelIndices.ToArray();
        }

        double SelectNextModelToRemove(ProbabilityPrediction[][] crossValidatedModelPredictions, double[] targets, double currentBestError)
        {
            var rows = crossValidatedModelPredictions.First().Length;
            var candidateModelMatrix = new ProbabilityPrediction[m_remainingModelIndices.Count - 1][];
            var candidatePredictions = new ProbabilityPrediction[rows];
            var candidateModelIndices = new int[m_remainingModelIndices.Count - 1];

            var bestError = currentBestError;
            var bestIndex = -1;

            foreach (var index in m_remainingModelIndices)
            {
                var candidateIndex = 0;
                for (int i = 0; i < m_remainingModelIndices.Count; i++)
                {
                    var curIndex = m_remainingModelIndices[i];
                    if (curIndex != index)
                    {
                        candidateModelIndices[candidateIndex++] = m_remainingModelIndices[i];
                    }
                }

                for (int i = 0; i < candidateModelIndices.Length; i++)
                {
                    candidateModelMatrix[i] = crossValidatedModelPredictions[candidateModelIndices[i]];
                }

                m_ensembleStrategy.Combine(candidateModelMatrix, candidatePredictions);
                var error = m_metric.Error(targets, candidatePredictions);

                if (error < bestError)
                {
                    bestError = error;
                    bestIndex = index;
                }
            }

            m_remainingModelIndices.Remove(bestIndex);

            return bestError;
        }
    }
}
