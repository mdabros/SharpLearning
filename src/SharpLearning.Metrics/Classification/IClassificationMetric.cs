
using System.Collections.Generic;
using SharpLearning.Common.Interfaces;
namespace SharpLearning.Metrics.Classification
{
    /// <summary>
    /// Classification metric interface
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IClassificationMetric<T> : IMetric<T, T>
    {
        /// <summary>
        /// Calculates the classification error
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        new double Error(T[] targets, T[] predictions);
        
        /// <summary>
        /// Gives a string representation of the classification matrix
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <returns></returns>
        string ErrorString(T[] targets, T[] predictions);

        /// <summary>
        /// Gives a string representation of the classification matrix.
        /// Using the target names provided in the targetStringMapping
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="targetStringMapping"></param>
        /// <returns></returns>
        string ErrorString(T[] targets, T[] predictions, Dictionary<T, string> targetStringMapping);
    }
}
