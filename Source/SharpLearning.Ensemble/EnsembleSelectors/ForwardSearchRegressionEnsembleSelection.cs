using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.Ensemble.Strategies;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Ensemble.EnsembleSelectors
{
    /// <summary>
    /// Greedy forward selection of ensemble models.
    /// </summary>
    public sealed class ForwardSearchRegressionEnsembleSelection : IRegressionEnsembleSelection
    {
        readonly IMetric<double, double> m_metric;
        readonly IRegressionEnsembleStrategy m_ensembleStrategy;
        readonly int m_numberOfModelsToSelect;
        readonly int m_numberOfModelsFromStart;
        readonly bool m_selectWithReplacement;
        List<int> m_remainingModelIndices;
        List<int> m_selectedModelIndices;

        /// <summary>
        /// Greedy forward selection of ensemble models.
        /// </summary>
        /// <param name="metric">Metric to minimize</param>
        /// <param name="ensembleStrategy">Strategy for ensembling models</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        /// <param name="numberOfModelsFromStart">Number of models from start of the search. 
        /// The top n models will be selected based in their solo performance</param>
        /// <param name="selectWithReplacement">If true the same model can be selected multiple times.
        /// This will correspond to weighting the models. If false each model can only be selected once</param>
        public ForwardSearchRegressionEnsembleSelection(IMetric<double, double> metric, IRegressionEnsembleStrategy ensembleStrategy,
            int numberOfModelsToSelect, int numberOfModelsFromStart, bool selectWithReplacement)
        {
            if (metric == null) { throw new ArgumentNullException("metric"); }
            if (ensembleStrategy == null) { throw new ArgumentNullException("ensembleStrategy"); }
            if (numberOfModelsToSelect < 1) { throw new ArgumentException("numberOfModelsToSelect must be at least 1"); }
            if (numberOfModelsFromStart < 1) { throw new ArgumentException("numberOfModelsFromStart must be at least 1"); }
            if (numberOfModelsFromStart > numberOfModelsToSelect) { throw new ArgumentException("numberOfModelsFromStart must be smaller than numberOfModelsToSelect"); }
            m_metric = metric;
            m_ensembleStrategy = ensembleStrategy;
            m_numberOfModelsToSelect = numberOfModelsToSelect;
            m_numberOfModelsFromStart = numberOfModelsFromStart;
            m_selectWithReplacement = selectWithReplacement;
        }

        /// <summary>
        /// Greedy forward selection of ensemble models.
        /// </summary>
        /// <param name="crossValidatedModelPredictions">cross validated predictions from multiple models. 
        /// Each column in the matrix corresponds to predictions from a separate model</param>
        /// <param name="targets">Corresponding targets</param>
        /// <returns>The indices of the selected model</returns>
        public int[] Select(F64Matrix crossValidatedModelPredictions, double[] targets)
        {
            if(crossValidatedModelPredictions.ColumnCount < m_numberOfModelsToSelect)
            {
                throw new ArgumentException("Availible models: " + crossValidatedModelPredictions.ColumnCount +
                    " is smaller than number of models to select: " + m_numberOfModelsToSelect);
            }

            var initialRanking = GetInitialRanking(crossValidatedModelPredictions, targets);

            m_selectedModelIndices = initialRanking
                .Take(m_numberOfModelsFromStart)
                .Select(v => v.Key).ToList();
            m_remainingModelIndices = initialRanking.Keys.ToList();

            if (!m_selectWithReplacement)
            {
                m_remainingModelIndices = m_remainingModelIndices
                    .Except(m_selectedModelIndices).ToList();
            }

            var currentError = double.MaxValue;

            for (int i = m_numberOfModelsFromStart; i < m_numberOfModelsToSelect; i++)
            {
                var error = SelectNextModelToAdd(crossValidatedModelPredictions, targets, currentError);

                if(error < currentError)
                {
                    currentError = error;
                    Trace.WriteLine("Models Selected: " + i + " Error: " + error);
                }
                else
                {
                    Trace.WriteLine("No error improvement. Stopping search");
                    break; // break when error does not improve
                }
            }

            Trace.WriteLine("Selected model indices: " + string.Join(", ", m_selectedModelIndices.ToArray()));

            return m_selectedModelIndices.ToArray();
        }

        double SelectNextModelToAdd(F64Matrix crossValidatedModelPredictions, double[] targets, double currentBestError)
        {
            var candidateModelMatrix = new F64Matrix(crossValidatedModelPredictions.RowCount, m_selectedModelIndices.Count + 1);
            var candidatePredictions = new double[crossValidatedModelPredictions.RowCount];
            var candidateModelIndices = new int[m_selectedModelIndices.Count + 1];

            var bestError = currentBestError;
            var bestIndex = -1;

            foreach (var index in m_remainingModelIndices)
            {
                m_selectedModelIndices.CopyTo(candidateModelIndices);
                candidateModelIndices[candidateModelIndices.Length - 1] = index;

                crossValidatedModelPredictions.Columns(candidateModelIndices, candidateModelMatrix);
                
                m_ensembleStrategy.Combine(candidateModelMatrix, candidatePredictions);
                var error = m_metric.Error(targets, candidatePredictions);

                if (error < bestError)
                {
                    bestError = error;
                    bestIndex = index;
                }
            }

            if(bestIndex != -1)
            {
                m_selectedModelIndices.Add(bestIndex);
                
                if(!m_selectWithReplacement)
                {
                    m_remainingModelIndices.Remove(bestIndex);
                }
            }

            return bestError;
        }

        Dictionary<int, double> GetInitialRanking(F64Matrix crossValidatedModelPredictions, double[] targets)
        {
            var ranking = new Dictionary<int, double>();
            var currentPredictions = new double[crossValidatedModelPredictions.RowCount];

            for (int i = 0; i < crossValidatedModelPredictions.ColumnCount; i++)
            {
                crossValidatedModelPredictions.Column(i, currentPredictions);
                var error = m_metric.Error(targets, currentPredictions);
                ranking.Add(i, error);
            }

            return ranking.OrderBy(v => v.Value).ToDictionary(v => v.Key, v => v.Value);
        }
    }
}
