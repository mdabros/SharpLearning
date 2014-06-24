using SharpLearning.Containers;
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
    public sealed class ClassificationCartModel
    {
        readonly IBinaryDecisionNode m_root;
        readonly double[] m_variableImportance;

        public ClassificationCartModel(IBinaryDecisionNode root, double[] variableImprotance)
        {
            if (root == null) { throw new ArgumentNullException("root"); }
            if (variableImprotance == null) { throw new ArgumentException("variableImprotance"); }
            m_root = root;
            m_variableImportance = variableImprotance;
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
                predictions[i] = m_root.Predict(observations.GetRow(indices[i]));
            }

            return predictions;
        }


        /// <summary>
        /// Predicts a single observation with probabilities
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            return m_root.PredictProbability(observation);
        }

        /// <summary>
        /// Predicts a set of observations with probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = m_root.PredictProbability(observations.GetRow(i));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts the observation subset provided by indices with probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations, int[] indices)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new ProbabilityPrediction[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                predictions[i] = m_root.PredictProbability(observations.GetRow(indices[i]));
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
