using SharpLearning.Containers.Matrices;

namespace SharpLearning.Ensemble.EnsembleSelectors;

/// <summary>
/// Interface for regression ensemble selection.
/// Finds the best subset of models to combine in an ensemble.
/// </summary>
public interface IRegressionEnsembleSelection
{
    /// <summary>
    /// Finds the best subset of models to combine in an ensemble.
    /// </summary>
    /// <param name="crossValidatedModelPredictions"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    int[] Select(F64Matrix crossValidatedModelPredictions, double[] targets);
}
