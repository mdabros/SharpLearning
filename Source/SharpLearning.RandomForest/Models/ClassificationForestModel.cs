using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.RandomForest.Models
{
    /// <summary>
    /// Classification forest model consiting of a series of decision trees
    /// </summary>
    public sealed class ClassificationForestModel
    {
        readonly ClassificationDecisionTreeModel[] m_models;
        readonly double[] m_rawVariableImportance;

        /// <summary>
        /// Classification forest model consiting of a series of decision trees
        /// </summary>
        /// <param name="models">The decision tree models</param>
        /// <param name="rawVariableImportance">The summed variable importance from all decision trees</param>
        public ClassificationForestModel(ClassificationDecisionTreeModel[] models, double[] rawVariableImportance)
        {
            if (models == null) { throw new ArgumentNullException("models"); }
            if (rawVariableImportance == null) { throw new ArgumentNullException("rawVariableImportance"); }
            m_models = models;
            m_rawVariableImportance = rawVariableImportance;
        }

        /// <summary>
        /// Predicts a single observations using majority vote
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = m_models.Select(m => m.Predict(observation))
                .GroupBy(p => p).OrderByDescending(g => g.Count())
                .First().Key;
            
            return prediction;
        }

        /// <summary>
        /// Predicts a set of obervations using majority vote
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

            foreach (var model in modelsProbability)
            {
                foreach (var probability in model)
                {
                    if(probabilities.ContainsKey(probability.Key))
                    {
                        probabilities[probability.Key] += probability.Value;
                    }
                    else
                    {
                        probabilities.Add(probability.Key, probability.Value);
                    }
                }
            }

            var keys = probabilities.Keys.ToList();
            foreach (var target in keys)
            {
                probabilities[target] /= m_models.Length;
            }

            var prediction = probabilities.OrderByDescending(p => p.Value)
                .First().Key;

            return new ProbabilityPrediction(prediction, probabilities);
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
                predictions[i] = PredictProbability(observations.GetRow(indices[i]));
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
