using System;
using System.Linq;

namespace SharpLearning.Ensemble.Strategies
{
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
    }
}
