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
    /// Classification forest model consisting of a series of decision trees
    /// </summary>
    [Serializable]
    public sealed class ClassificationForestModel : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        readonly double[] m_rawVariableImportance;

        /// <summary>
        /// Classification forest model consisting of a series of decision trees
        /// </summary>
        /// <param name="models">The decision tree models</param>
        /// <param name="rawVariableImportance">The summed variable importance from all decision trees</param>
        public ClassificationForestModel(ClassificationDecisionTreeModel[] models, double[] rawVariableImportance)
        {
            Trees = models ?? throw new ArgumentNullException("models");
            m_rawVariableImportance = rawVariableImportance ?? throw new ArgumentNullException("rawVariableImportance");
        }

        /// <summary>
        /// Individual trees from the ensemble.
        /// </summary>
        public ClassificationDecisionTreeModel[] Trees { get; }

        /// <summary>
        /// Predicts a single observations using majority vote
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = Trees.Select(m => m.Predict(observation))
                .GroupBy(p => p).OrderByDescending(g => g.Count())
                .First().Key;
            
            return prediction;
        }

        /// <summary>
        /// Predicts a set of observations using majority vote
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
        /// Predicts a single observation using the ensembled probabilities
        /// Note this can yield a different result than using regular predict
        /// Usually this will be a more accurate predictions
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var probabilities = new Dictionary<double, double>();
            var modelsProbability = Trees.Select(m => m.PredictProbability(observation).Probabilities)
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
                probabilities[target] /= Trees.Length;
            }

            var prediction = probabilities.OrderByDescending(p => p.Value)
                .First().Key;

            return new ProbabilityPrediction(prediction, probabilities);
        }

        /// <summary>
        /// Predicts a set of observations using the ensembled probabilities
        /// Note this can yield a different result than using regular predict
        /// Usually this will be a more accurate predictions
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = PredictProbability(observations.Row(i));
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
        /// Loads a ClassificationForestModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static ClassificationForestModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<ClassificationForestModel>(reader);
        }

        /// <summary>
        /// Saves the ClassificationForestModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }

        /// <summary>
        /// Private explicit interface implementation for probability predictions
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        ProbabilityPrediction IPredictor<ProbabilityPrediction>.Predict(double[] observation) 
            => PredictProbability(observation);

        /// <summary>
        /// Private explicit interface implementation for probability predictions
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        ProbabilityPrediction[] IPredictor<ProbabilityPrediction>.Predict(F64Matrix observations) 
            => PredictProbability(observations);
    }
}
