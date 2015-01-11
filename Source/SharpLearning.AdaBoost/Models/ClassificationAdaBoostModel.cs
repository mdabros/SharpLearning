using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.AdaBoost.Models
{
    /// <summary>
    /// AdaBoost classification model. Consist of a series of tree model and corresponding weights
    /// </summary>
    [Serializable]
    public sealed class ClassificationAdaBoostModel : IPredictor<double>, IPredictor<ProbabilityPrediction>
    {
        readonly double[] m_modelWeights;
        readonly ClassificationDecisionTreeModel[] m_models;
        readonly double[] m_rawVariableImportance;
        Dictionary<double, double> m_predictions = new Dictionary<double, double>();

        /// <summary>
        /// AdaBoost classification model. Consist of a series of tree model and corresponding weights
        /// </summary>
        /// <param name="models"></param>
        /// <param name="modelWeights"></param>
        /// <param name="rawVariableImportance"></param>
        public ClassificationAdaBoostModel(ClassificationDecisionTreeModel[] models, double[] modelWeights,
            double[] rawVariableImportance)
        {
            if (models == null) { throw new ArgumentNullException("models"); }
            if (modelWeights == null) { throw new ArgumentNullException("modelWeights"); }
            if (rawVariableImportance == null) { throw new ArgumentNullException("rawVariableImportance"); }

            m_models = models;
            m_modelWeights = modelWeights;
            m_rawVariableImportance = rawVariableImportance;
        }

        /// <summary>
        /// Predicts a single observations using weighted majority vote
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var count = m_models.Length;
            m_predictions.Clear();

            for (int i = 0; i < count; i++)
            {
                var prediction = m_models[i].Predict(observation);
                var weight = m_modelWeights[i];

                if (m_predictions.ContainsKey(prediction))
                {
                    m_predictions[prediction] += weight;
                }
                else
                {
                    m_predictions.Add(prediction, weight);
                }
            }

            return m_predictions.OrderByDescending(v => v.Value).First().Key;
        }

        /// <summary>
        /// Predicts a set of obervations using weighted majority vote
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
        /// Predicts a single observation using the ensembled probabilities
        /// Note this can yield a different result than using regular predict
        /// Usally this will be a more accurate predictions
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var probabilities = new Dictionary<double, double>();
            var modelsProbability = m_models.Select(m => m.PredictProbability(observation).Probabilities)
                .ToArray();

            for (int i = 0; i < modelsProbability.Length; i++)
            {
                var model = modelsProbability[i];
                var w = m_modelWeights[i];

                foreach (var probability in model)
                {
                    if (probabilities.ContainsKey(probability.Key))
                    {
                        probabilities[probability.Key] += w * probability.Value;
                    }
                    else
                    {
                        probabilities.Add(probability.Key, w * probability.Value);
                    }
                }
            }

            var keys = probabilities.Keys.ToList();
            var probabilityFactor = 1.0 / m_modelWeights.Sum();

            foreach (var target in keys)
            {
                probabilities[target] *= probabilityFactor;
            }

            var prediction = probabilities.OrderByDescending(p => p.Value)
                .First().Key;

            return new ProbabilityPrediction(prediction, probabilities);
        }

        /// <summary>
        /// Private explicit interface implementation for probability predictions
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        ProbabilityPrediction IPredictor<ProbabilityPrediction>.Predict(double[] observation)
        {
            return PredictProbability(observation);
        }

        /// <summary>
        /// Predicts a set of obervations using the ensembled probabilities
        /// Note this can yield a different result than using regular predict
        /// Usally this will be a more accurate predictions
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = PredictProbability(observations.GetRow(i));
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
        /// Loads a ClassificationAdaBoostModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static ClassificationAdaBoostModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<ClassificationAdaBoostModel>(reader);
        }

        /// <summary>
        /// Saves the ClassificationAdaBoostModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }
    }
}
