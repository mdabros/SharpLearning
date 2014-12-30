using SharpLearning.Containers;
using SharpLearning.Learners.Interfaces;

namespace SharpLearning.Metrics.Classification
{
    /// <summary>
    /// Classification probability metric interface
    /// </summary>
    public interface IClassificationProbabilityMetric : IMetric<double, ProbabilityPrediction>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        new double Error(double[] targets, ProbabilityPrediction[] predictions);
     }
}
