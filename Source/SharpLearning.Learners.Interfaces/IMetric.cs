
namespace SharpLearning.Learners.Interfaces
{
    /// <summary>
    /// Generel metric interface
    /// </summary>
    /// <typeparam name="IPrediction"></typeparam>
    public interface IMetric<IPrediction>
    {
        /// <summary>
        /// Returns an error metric based on the targets and predictions
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        double Error(IPrediction[] targets, IPrediction[] predictions);
    }
}
