using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Views;
using SharpLearning.Ensemble.Strategies;
using System;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Ensemble.EnsembleSelectors
{
    /// <summary>
    /// Iterative random selection of ensemble models.
    /// </summary>
    public sealed class RandomClassificationEnsembleSelection : IClassificationEnsembleSelection
    {
        readonly IMetric<double, ProbabilityPrediction> m_metric;
        readonly IClassificationEnsembleStrategy m_ensembleStrategy;
        readonly int m_numberOfModelsToSelect;
        readonly int m_iterations;
        readonly bool m_selectWithReplacement;
        readonly Random m_random;
        int[] m_allIndices;

        /// <summary>
        /// Iterative random selection of ensemble models.
        /// </summary>
        /// <param name="metric">Metric to minimize</param>
        /// <param name="ensembleStrategy">Strategy for ensembling models</param>
        /// <param name="numberOfModelsToSelect">Number of models to select</param>
        /// <param name="iterations">Number of iterations to try random selection</param>
        /// <param name="selectWithReplacement">If true the same model can be selected multiple times.
        /// This will correspond to weighting the models. If false each model can only be selected once</param>
        /// <param name="seed"></param>
        public RandomClassificationEnsembleSelection(IMetric<double, ProbabilityPrediction> metric, IClassificationEnsembleStrategy ensembleStrategy,
            int numberOfModelsToSelect, int iterations, bool selectWithReplacement, int seed = 42)
        {
            if (metric == null) { throw new ArgumentNullException("metric"); }
            if (ensembleStrategy == null) { throw new ArgumentNullException("ensembleStrategy"); }
            if (numberOfModelsToSelect < 1) { throw new ArgumentException("numberOfModelsToSelect must be at least 1"); }
            if (iterations < 1) { throw new ArgumentException("Number of iterations"); }

            m_metric = metric;
            m_ensembleStrategy = ensembleStrategy;
            m_numberOfModelsToSelect = numberOfModelsToSelect;
            m_selectWithReplacement = selectWithReplacement;
            m_iterations = iterations;
            m_random = new Random(seed);
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

            m_allIndices = Enumerable.Range(0, crossValidatedModelPredictions.Length).ToArray();

            var rows = crossValidatedModelPredictions.First().Length;
            var candidateModelMatrix = new ProbabilityPrediction[m_numberOfModelsToSelect][];
            var candidatePredictions = new ProbabilityPrediction[rows];
            var candidateModelIndices = new int[m_numberOfModelsToSelect];
            var bestModelIndices = new int[m_numberOfModelsToSelect];

            var bestError = double.MaxValue;

            for (int i = 0; i < m_iterations; i++)
            {
                SelectNextRandomIndices(candidateModelIndices);

                for (int j = 0; j < candidateModelIndices.Length; j++)
                {
                    candidateModelMatrix[j] = crossValidatedModelPredictions[candidateModelIndices[j]];
                }

                m_ensembleStrategy.Combine(candidateModelMatrix, candidatePredictions);
                var error = m_metric.Error(targets, candidatePredictions);

                if (error < bestError)
                {
                    bestError = error;
                    candidateModelIndices.CopyTo(bestModelIndices, 0);
                    Trace.WriteLine("Models selected: " + bestModelIndices.Length+ ": " + error);
                }
            }

            Trace.WriteLine("Selected model indices: " + string.Join(", ", bestModelIndices.ToArray()));

            return bestModelIndices;
        }

        void SelectNextRandomIndices(int[] candidateModelIndices)
        {
            if(m_selectWithReplacement)
            {
                for (int i = 0; i < candidateModelIndices.Length; i++)
                {
                    candidateModelIndices[i] = m_random.Next(0, m_numberOfModelsToSelect);
                }
            }
            else
            {
                m_allIndices.Shuffle(m_random);
                m_allIndices.CopyTo(Interval1D.Create(0, m_numberOfModelsToSelect), candidateModelIndices);
            }
        }
    }
}
