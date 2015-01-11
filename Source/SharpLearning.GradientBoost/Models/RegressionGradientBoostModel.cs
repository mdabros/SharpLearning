using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.GradientBoost.LossFunctions;
using SharpLearning.Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.GradientBoost.Models
{
    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public sealed class RegressionGradientBoostModel : IPredictor<double>
    {
        readonly RegressionDecisionTreeModel[] m_models;
        readonly double[] m_rawVariableImportance;
        readonly double m_learningRate;
        readonly double m_initialLoss;
        readonly double[] m_predictions;

        public RegressionGradientBoostModel(RegressionDecisionTreeModel[] models, double[] rawVariableImportance, 
            double learningRate, double initialLoss)
        {
            if (models == null) { throw new ArgumentNullException("models"); }
            if (rawVariableImportance == null) { throw new ArgumentNullException("rawVariableImportance"); }
            if (learningRate <= 0.0) { throw new ArgumentException("learning rate must be larger than 0"); }
            m_models = models;
            m_rawVariableImportance = rawVariableImportance;
            m_learningRate = learningRate;
            m_initialLoss = initialLoss;

            m_predictions = new double[models.Length];
        }

        /// <summary>
        /// Predicts a single observations using the combination of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = m_initialLoss;

            foreach (var model in m_models)
            {
                prediction += m_learningRate * model.Predict(observation);
            }

            return prediction;
        }

        /// <summary>
        /// Predicts a set of obervations using the combination of all predictors
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

        /// <summary>
        /// Loads a RegressionGradientBoostModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static RegressionGradientBoostModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<RegressionGradientBoostModel>(reader);
        }

        /// <summary>
        /// Saves the RegressionGradientBoostModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize<RegressionGradientBoostModel>(this, writer);
        }
    }
}
