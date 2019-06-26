using System.Collections.Generic;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;

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

        /// <summary>
        /// Gives a string representation of the classification matrix
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        string ErrorString(double[] targets, ProbabilityPrediction[] predictions);

        /// <summary>
        /// Gives a string representation of the classification matrix.
        /// Using the target names provided in the targetStringMapping
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="targetStringMapping"></param>
        /// <returns></returns>
        string ErrorString(double[] targets, ProbabilityPrediction[] predictions, 
            Dictionary<double, string> targetStringMapping);
     }
}
