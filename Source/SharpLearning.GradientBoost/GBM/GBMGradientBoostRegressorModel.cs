using SharpLearning.Containers.Matrices;
using System;
using System.Linq;
using System.Collections.Generic;
using SharpLearning.Common.Interfaces;
using SharpLearning.InputOutput.Serialization;
using System.IO;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public sealed class GBMGradientBoostRegressorModel : IPredictor<double>
    {
        readonly GBMTree[] m_trees;
        readonly double m_learningRate;
        readonly double m_initialLoss;
        readonly int m_featureCount;

        public GBMGradientBoostRegressorModel(GBMTree[] trees, double learningRate, double initialLoss, int featureCount)
        {
            if (trees == null) { throw new ArgumentNullException("trees"); }
            m_trees = trees;
            m_learningRate = learningRate;
            m_initialLoss = initialLoss;
            m_featureCount = featureCount;
        }

        /// <summary>
        /// Predicts a single observations using the combination of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = m_initialLoss;
            for (int i = 0; i < m_trees.Length; i++)
            {
                prediction += m_learningRate * m_trees[i].Predict(observation);
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
            var m_rawVariableImportance = GetRawVariableImportance();
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
            var rawVariableImportance = new double[m_featureCount];
            foreach (var tree in m_trees)
            {
                tree.AddRawVariableImportances(rawVariableImportance);
            }

            return rawVariableImportance;
        }

        /// <summary>
        /// Loads a GBMGradientBoostRegressorModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static GBMGradientBoostRegressorModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<GBMGradientBoostRegressorModel>(reader);
        }

        /// <summary>
        /// Saves the GBMGradientBoostRegressorModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }
    }
}
