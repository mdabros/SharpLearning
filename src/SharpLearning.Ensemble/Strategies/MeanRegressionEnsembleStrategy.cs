using System;
using System.Linq;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Ensemble.Strategies;

/// <summary>
/// Mean regression ensemble strategy. Models are combined using standard mean.
/// </summary>
[Serializable]
public sealed class MeanRegressionEnsembleStrategy : IRegressionEnsembleStrategy
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="ensemblePredictions"></param>
    /// <returns></returns>
    public double Combine(double[] ensemblePredictions)
    {
        return ensemblePredictions.Average();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="ensemblePredictions"></param>
    /// <param name="predictions"></param>
    public void Combine(F64Matrix ensemblePredictions, double[] predictions)
    {
        var cols = ensemblePredictions.ColumnCount;
        var rows = ensemblePredictions.RowCount;

        for (int i = 0; i < rows; i++)
        {
            var sum = 0.0;
            for (int j = 0; j < cols; j++)
            {
                sum += ensemblePredictions.At(i, j);
            }
            predictions[i] = sum / (double)cols;
        }
    }
}
