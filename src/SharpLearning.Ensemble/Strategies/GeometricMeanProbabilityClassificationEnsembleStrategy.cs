using System;
using System.Linq;
using SharpLearning.Containers;

namespace SharpLearning.Ensemble.Strategies;

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

    /// <summary>
    /// Geometric mean probability classification ensemble strategy. Class probabilities are combined using the geometric mean across all models.
    /// </summary>
    /// <param name="ensemblePredictions"></param>
    /// <param name="predictions"></param>
    public void Combine(ProbabilityPrediction[][] ensemblePredictions, ProbabilityPrediction[] predictions)
    {
        var currentObservation = new ProbabilityPrediction[ensemblePredictions.Length];

        for (int i = 0; i < predictions.Length; i++)
        {
            for (int j = 0; j < currentObservation.Length; j++)
            {
                currentObservation[j] = ensemblePredictions[j][i];
            }
            predictions[i] = Combine(currentObservation);
        }
    }

    static double GeometricMean(double[] values)
    {
        var geoMean = 0.0;
        for (int i = 0; i < values.Length; i++)
        {
            if (i == 0)
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
