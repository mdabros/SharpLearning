using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.GradientBoost.LossFunctions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.GradientBoost.Models
{
    public sealed class RegressionGradientBoostModel
    {
        readonly RegressionDecisionTreeModel[] m_models;
        readonly double[] m_rawVariableImportance;
        readonly ILossFunction m_lossFunction;
        readonly double[] m_predictions;

        public RegressionGradientBoostModel(RegressionDecisionTreeModel[] models, double[] rawVariableImportance, ILossFunction lossFunction)
        {
            if (models == null) { throw new ArgumentNullException("models"); }
            if (rawVariableImportance == null) { throw new ArgumentNullException("rawVariableImportance"); }
            if (lossFunction == null) { throw new ArgumentNullException("lossFunction"); }

            m_models = models;
            m_rawVariableImportance = rawVariableImportance;
            m_lossFunction = lossFunction;

            m_predictions = new double[models.Length];
        }

        /// <summary>
        /// Predicts a single observations using the mean of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = m_lossFunction.InitialLoss;

            foreach (var model in m_models)
            {
                prediction += m_lossFunction.LearningRate * model.Predict(observation);
            }

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
                predictions[i] = Predict(observations.GetRow(indices[i]));
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
