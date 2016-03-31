using SharpLearning.Containers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Ensemble.Strategies
{
    /// <summary>
    /// Mean probability classification ensemble strategy. Class probabilities are combined using the mean across all models.
    /// </summary>
    [Serializable]
    public sealed class MeanProbabilityClassificationEnsembleStrategy : IClassificationEnsembleStrategy
    {
        /// <summary>
        /// Mean probability classification ensemble strategy. Class probabilities are combined using the mean across all models.
        /// </summary>
        /// <param name="ensemblePredictions"></param>
        /// <returns></returns>
        public ProbabilityPrediction Combine(ProbabilityPrediction[] ensemblePredictions)
        {
            var averageProbabilities = ensemblePredictions.Select(p => p.Probabilities).SelectMany(d => d)
                         .GroupBy(kvp => kvp.Key)
                         .ToDictionary(g => g.Key, g => g.Average(kvp => kvp.Value));

            var prediction = averageProbabilities.OrderByDescending(d => d.Value).First().Key;

            return new ProbabilityPrediction(prediction, averageProbabilities);
        }
    }
}
