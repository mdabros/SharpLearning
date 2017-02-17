using SharpLearning.Common.Interfaces;

namespace SharpLearning.Metrics.Ranking
{
    /// <summary>
    /// Ranking metric interface
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IRankingMetric<T> : IMetric<T, T>
    {
        /// <summary>
        /// Calculates the ranking error
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        new double Error(T[] targets, T[] predictions);
    }
}
