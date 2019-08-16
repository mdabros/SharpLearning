
namespace SharpLearning.Common.Interfaces
{
    /// <summary>
    /// General metric interface
    /// </summary>
    /// <typeparam name="TTarget"></typeparam>
    /// <typeparam name="TPrediction"></typeparam>
    public interface IMetric<TTarget, TPrediction>
    {
        /// <summary>
        /// Returns an error metric based on the targets and predictions
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        double Error(TTarget[] targets, TPrediction[] predictions);
    }
}
