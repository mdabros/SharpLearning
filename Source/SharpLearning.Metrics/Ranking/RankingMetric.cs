
namespace SharpLearning.Metrics.Ranking
{
    /// <summary>
    /// Ranking metric interface
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface RankingMetric<T>
    {
        /// <summary>
        /// Calculates the ranking error
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        double Error(T[] targets, T[] predictions);
    }
}
