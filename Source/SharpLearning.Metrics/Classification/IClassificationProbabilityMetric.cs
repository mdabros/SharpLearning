using SharpLearning.Containers;

namespace SharpLearning.Metrics.Classification
{
    /// <summary>
    /// Classification probability metric interface
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IClassificationProbabilityMetric
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        double Error(double[] targets, ProbabilityPrediction[] predictions);
     }
}
