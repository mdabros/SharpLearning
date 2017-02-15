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
    /// Greedy backward elimination of ensemble models.
    /// </summary>
    public sealed class BackwardEliminationRegressionEnsembleSelection : IRegressionEnsembleSelection
    {
        readonly IMetric<double, double> m_metric;
        readonly IRegressionEnsembleStrategy m_ensembleStrategy;
        readonly int m_numberOfModelsToSelect;
        List<int> m_remainingModelIndices;
        List<int> m_bestModelIndices;


        /// <summary>
        /// Greedy backward elimination of ensemble models.
        /// </summary>
        /// <param name="metric">Metric to minimize</param>
        /// <param name="ensembleStrategy">Strategy for ensembling models</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        public BackwardEliminationRegressionEnsembleSelection(IMetric<double, double> metric, IRegressionEnsembleStrategy ensembleStrategy,
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
        /// Greedy backward elimination of ensemble models.
        /// </summary>
        /// <param name="crossValidatedModelPredictions">cross validated predictions from multiple models. 
        /// Each column in the matrix corresponds to predictions from a separate model</param>
        /// <param name="targets">Corresponding targets</param>
        /// <returns>The indices of the selected model</returns>
        public int[] Select(F64Matrix crossValidatedModelPredictions, double[] targets)
        {
            if(crossValidatedModelPredictions.ColumnCount() < m_numberOfModelsToSelect)
            {
                throw new ArgumentException("Availible models: " + crossValidatedModelPredictions.ColumnCount() +
                    " is smaller than number of models to select: " + m_numberOfModelsToSelect);
            }

            m_remainingModelIndices = Enumerable.Range(0, crossValidatedModelPredictions.ColumnCount()).ToList();

            var currentError = double.MaxValue;
            var modelsToRemove = m_remainingModelIndices.Count - 1;

            for (int i = 0; i < modelsToRemove; i++)
            {
                var error = SelectNextModelToRemove(crossValidatedModelPredictions, targets, currentError);

                if(error < currentError && m_remainingModelIndices.Count <= m_numberOfModelsToSelect)
                {
                    currentError = error;
                    m_bestModelIndices = m_remainingModelIndices.ToList();
                    Trace.WriteLine("Models selected: " + m_bestModelIndices.Count + ": " + error);
                }
            }

                Trace.WriteLine("Selected model indices: " + string.Join(", ", m_bestModelIndices.ToArray()));

            return m_bestModelIndices.ToArray();
        }

        double SelectNextModelToRemove(F64Matrix crossValidatedModelPredictions, double[] targets, double currentBestError)
        {
            var candidateModelMatrix = new F64Matrix(crossValidatedModelPredictions.RowCount(), m_remainingModelIndices.Count - 1);
            var candidatePredictions = new double[crossValidatedModelPredictions.RowCount()];
            var candidateModelIndices = new int[m_remainingModelIndices.Count - 1];

            var bestError = double.MaxValue;
            var bestIndex = -1;

            foreach (var index in m_remainingModelIndices)
            {
                var candidateIndex = 0;
                for (int i = 0; i < m_remainingModelIndices.Count; i++)
                {
                    var curIndex = m_remainingModelIndices[i];
                    if(curIndex != index)
                    {
                        candidateModelIndices[candidateIndex++] = m_remainingModelIndices[i];
                    }
                }

                crossValidatedModelPredictions.Columns(candidateModelIndices, candidateModelMatrix);
                
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
