using System.Collections.Generic;

namespace SharpLearning.Common.Interfaces
{
    /// <summary>
    /// Interface for variable importance
    /// </summary>
    public interface IModelVariableImportance
    {
        /// <summary>
        /// Gets the raw unsorted variable importance scores
        /// </summary>
        /// <returns></returns>
        double[] GetRawVariableImportance();

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex);
    }
}
