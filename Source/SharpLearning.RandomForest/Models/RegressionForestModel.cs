using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.InputOutput.Serialization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.RandomForest.Models
{
    /// <summary>
    /// Regression forest model consiting of a series of decision trees
    /// </summary>
    [Serializable]
    public sealed class RegressionForestModel : IPredictorModel<double>
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
            var rows = observations.RowCount();
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = Predict(observations.GetRow(i));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts a single observation and provides a certainty measure on the prediction
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public CertaintyPrediction PredictCertainty(double[] observation)
        {
            var prediction = Predict(observation);
            var variance = 0.0;
            for (int i = 0; i < m_models.Length; i++)
            {
                var temp = m_models[i].Predict(observation) - prediction;
                variance += temp * temp;
            }

            variance = variance / (double)m_models.Length;

            return new CertaintyPrediction(prediction, variance);
        }

        /// <summary>
        /// Predicts a set of obervations with certainty predictions
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public CertaintyPrediction[] PredictCertainty(F64Matrix observations)
        {
            var rows = observations.RowCount();
            var predictions = new CertaintyPrediction[rows];

            for (int i = 0; i < rows; i++)
            {
                predictions[i] = PredictCertainty(observations.GetRow(i));
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
        /// Loads a RegressionForestModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static RegressionForestModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<RegressionForestModel>(reader);
        }

        /// <summary>
        /// Saves the RegressionForestModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }
    }
}
