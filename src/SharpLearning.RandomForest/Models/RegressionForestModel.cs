using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.RandomForest.Models
{
    /// <summary>
    /// Regression forest model consisting of a series of decision trees
    /// </summary>
    [Serializable]
    public sealed class RegressionForestModel : IPredictorModel<double>
    {
        readonly double[] m_rawVariableImportance;

        /// <summary>
        /// Classification forest model consisting of a series of decision trees
        /// </summary>
        /// <param name="trees">The decision tree models</param>
        /// <param name="rawVariableImportance">The summed variable importance from all decision trees</param>
        public RegressionForestModel(RegressionDecisionTreeModel[] trees, double[] rawVariableImportance)
        {
            Trees = trees ?? throw new ArgumentNullException("models");
            m_rawVariableImportance = rawVariableImportance ?? throw new ArgumentNullException("rawVariableImportance");
        }

        /// <summary>
        /// Individual trees from the ensemble.
        /// </summary>
        public RegressionDecisionTreeModel[] Trees { get; }

        /// <summary>
        /// Predicts a single observations using the mean of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = Trees.Select(m => m.Predict(observation))
                .Average();
            
            return prediction;
        }

        /// <summary>
        /// Predicts a set of observations using the mean of all predictors
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = Predict(observations.Row(i));
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
            for (int i = 0; i < Trees.Length; i++)
            {
                var temp = Trees[i].Predict(observation) - prediction;
                variance += temp * temp;
            }

            variance = variance / (double)Trees.Length;

            return new CertaintyPrediction(prediction, variance);
        }

        /// <summary>
        /// Predicts a set of observations with certainty predictions
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public CertaintyPrediction[] PredictCertainty(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new CertaintyPrediction[rows];

            for (int i = 0; i < rows; i++)
            {
                predictions[i] = PredictCertainty(observations.Row(i));
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
        /// Gets the raw unsorted variable importance scores
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance() => m_rawVariableImportance;

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
