using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.RandomForest.Models
{
    /// <summary>
    /// Regression forest model consiting of a series of decision trees
    /// </summary>
    public sealed class RegressionForestModel
    {
        readonly RegressionDecisionTreeModel[] m_models;
        readonly double[] m_rawVariableImportance;

        /// <summary>
        /// Classification forest model consiting of a series of decision trees
        /// </summary>
        /// <param name="models">The decision tree models</param>
        /// <param name="rawVariableImportance">The summed variable importance from all decision trees</param>
        public RegressionForestModel(RegressionDecisionTreeModel[] models, double[] rawVariableImportance)
        {
            if (models == null) { throw new ArgumentNullException("models"); }
            if (rawVariableImportance == null) { throw new ArgumentNullException("rawVariableImportance"); }
            m_models = models;
            m_rawVariableImportance = rawVariableImportance;
        }

        /// <summary>
        /// Predicts a single observations using the mean of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = m_models.Select(m => m.Predict(observation))
                .Average();
            
            return prediction;
        }

        /// <summary>
        /// Predicts a set of obervations using the mean of all predictors
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = Predict(observations.GetRow(i));
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
            var max = m_rawVariableImportance.Max();

            var scaledVariableImportance = m_rawVariableImportance
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
            return m_rawVariableImportance;
        }
    }
}
