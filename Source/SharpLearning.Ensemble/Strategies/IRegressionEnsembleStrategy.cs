using SharpLearning.Containers.Matrices;

namespace SharpLearning.Ensemble.Strategies
{
    /// <summary>
    /// Interface for regression ensemble strategies
    /// </summary>
    public interface IRegressionEnsembleStrategy
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="ensemblePredictions"></param>
        /// <returns></returns>
        double Combine(double[] ensemblePredictions);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ensemblePredictions"></param>
        /// <param name="predictions"></param>
        void Combine(F64Matrix ensemblePredictions, double[] predictions);
    }
}
