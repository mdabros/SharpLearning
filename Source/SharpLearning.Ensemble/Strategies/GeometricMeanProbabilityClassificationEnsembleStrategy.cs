using SharpLearning.Containers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Ensemble.Strategies
{
    /// <summary>
    /// Geometric mean probability classification ensemble strategy. Class probabilities are combined using the geometric mean across all models.
    /// https://en.wikipedia.org/wiki/Geometric_mean
    /// </summary>
    [Serializable]
    public sealed class GeometricMeanProbabilityClassificationEnsembleStrategy : IClassificationEnsembleStrategy
    {
        /// <summary>
        /// Geometric mean probability classification ensemble strategy. Class probabilities are combined using the geometric mean across all models.
        /// </summary>
        /// <param name="ensemblePredictions"></param>
        /// <returns></returns>
        public ProbabilityPrediction Combine(ProbabilityPrediction[] ensemblePredictions)
        {
            var averageProbabilities = ensemblePredictions.Select(p => p.Probabilities).SelectMany(d => d)
                         .GroupBy(kvp => kvp.Key)
                         .ToDictionary(g => g.Key, g => GeometricMean(g.Select(p => p.Value).ToArray()));

            var sum = averageProbabilities.Values.Sum();
            averageProbabilities = averageProbabilities.ToDictionary(p => p.Key, p => p.Value / sum);
            
            var prediction = averageProbabilities.OrderByDescending(d => d.Value).First().Key;

            return new ProbabilityPrediction(prediction, averageProbabilities);
        }

        double GeometricMean(double[] values)
        {
            var geoMean = 0.0;
            for (int i = 0; i < values.Length; i++)
            {
                if(i == 0)
                {
                    geoMean = values[i];
                }
                else
                {
                    geoMean *= values[i];

                }
            }

            return Math.Pow(geoMean, 1.0 / (double)values.Length);
        }
    }
}
