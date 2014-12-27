using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Learners.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.DecisionTrees.Models
{
    /// <summary>
    /// Regression Decision tree model
    /// </summary>
    public sealed class RegressionDecisionTreeModel : IPredictor<double>
    {
        public readonly BinaryTree Tree;
        readonly double[] m_variableImportance;

        public RegressionDecisionTreeModel(BinaryTree tree)
        {
            if (tree == null) { throw new ArgumentNullException("root"); }
            Tree = tree;
            m_variableImportance = Tree.VariableImportance;
        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return Tree.Predict(observation);
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
                predictions[i] = Tree.Predict(observations.GetRow(i));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts the observation subset provided by indices
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations, int[] indices)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new double[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                predictions[i] = Tree.Predict(observations.GetRow(indices[i]));
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
