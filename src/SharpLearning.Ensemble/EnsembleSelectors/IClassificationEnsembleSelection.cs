using SharpLearning.Containers;

namespace SharpLearning.Ensemble.EnsembleSelectors
{
    /// <summary>
    /// Interface for classification ensemble selection.
    /// Finds the best subset of models to combine in an ensemble.
    /// </summary>
    public interface IClassificationEnsembleSelection
    {
        /// <summary>
        /// Finds the best subset of models to combine in an ensemble.
        /// </summary>
        /// <param name="crossValidatedModelPredictions"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        int[] Select(ProbabilityPrediction[][] crossValidatedModelPredictions, double[] targets);
    }
}
