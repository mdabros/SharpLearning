using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.DecisionTrees.Models
{
    /// <summary>
    /// CART Decision tree model
    /// </summary>
    public sealed class RegressionCartModel
    {
        readonly IBinaryDecisionNode m_root;
        readonly double[] m_variableImportance;

        public RegressionCartModel(IBinaryDecisionNode root, double[] variableImportance)
        {
            if (root == null) { throw new ArgumentNullException("root"); }
            if (variableImportance == null) { throw new ArgumentException("variableImportance"); }
            m_root = root;
            m_variableImportance = variableImportance;
        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return m_root.Predict(observation);
        }

        /// <summary>
        /// Predicts a set of observations 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = m_root.Predict(observations.GetRow(i));
            }

            return predictions;
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var max = m_variableImportance.Max();

            var scaledVariableImportance = m_variableImportance
                .Select(v => (v / max) * 100.0)
                .ToArray();

            return featureNameToIndex.ToDictionary(kvp => kvp.Key, kvp => scaledVariableImportance[kvp.Value])
                        .OrderByDescending(kvp => kvp.Value)
                        .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }

        /// <summary>
        /// Gets the raw unsorted vatiable importance scores
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            return m_variableImportance;
        }
    }
}
