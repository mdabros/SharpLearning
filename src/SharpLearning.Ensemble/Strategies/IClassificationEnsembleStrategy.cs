using SharpLearning.Containers;

namespace SharpLearning.Ensemble.Strategies
{
    /// <summary>
    /// Interface for classification ensemble strategies
    /// </summary>
    public interface IClassificationEnsembleStrategy
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="ensemblePredictions"></param>
        /// <returns></returns>
        ProbabilityPrediction Combine(ProbabilityPrediction[] ensemblePredictions);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ensemblePredictions"></param>
        /// <param name="predictions"></param>
        void Combine(ProbabilityPrediction[][] ensemblePredictions, ProbabilityPrediction[] predictions);

    }
}
