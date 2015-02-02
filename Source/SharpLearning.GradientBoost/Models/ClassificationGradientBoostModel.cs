using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
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
    /// Classification gradient boost model
    /// </summary>
    [Serializable]
    public sealed class ClassificationGradientBoostModel : IPredictor<double>, IPredictor<ProbabilityPrediction>
    {
        readonly RegressionDecisionTreeModel[][] m_models;
        readonly double[] m_rawVariableImportance;
        readonly double m_learningRate;
        readonly double[] m_priorPrababilities;
        readonly double[] m_predictions;
        readonly double[] m_targetNames;

        public ClassificationGradientBoostModel(RegressionDecisionTreeModel[][] models, double[] rawVariableImportance, 
            double learningRate, double[] priorPrababilities, double[] targetNames)
        {
            if (models == null) { throw new ArgumentNullException("models"); }
            if (rawVariableImportance == null) { throw new ArgumentNullException("rawVariableImportance"); }
            if (learningRate <= 0.0) { throw new ArgumentException("learning rate must be larger than 0"); }
            if (priorPrababilities == null) { throw new ArgumentException("priorPrababilities"); }
            if (targetNames == null) { throw new ArgumentException("targetNames"); }
            m_models = models;
            m_rawVariableImportance = rawVariableImportance;
            m_learningRate = learningRate;
            m_priorPrababilities = priorPrababilities;
            m_targetNames = targetNames;

            m_predictions = new double[models.Length];
        }

        double[] Scores(double[] observation)
        {
            var scores = m_priorPrababilities.ToArray();

            for (int i = 0; i < m_models.Length; i++)
            {
                var targetModels = m_models[i];
                foreach (var model in targetModels)
                {
                    scores[i] += m_learningRate * model.Predict(observation);
                }
            }

            return scores;
        }

        ProbabilityPrediction ScoresToProbability(double[] scores)
        {
            var probabilities = new Dictionary<double, double>();
            var expScores = scores.Select(s => Math.Exp(s))
                .ToArray();

            var maxPrediction = m_targetNames.First();
            var maxProbability = -1.0;

            for (int i = 0; i < scores.Length; i++)
            {
                var targetName = m_targetNames[i];
                var score = scores[i];
                var probability = Math.Exp(score - Math.Log(expScores.Sum()))
                    .NanToNum();

                if(probability > maxProbability)
                {
                    maxProbability = probability;
                    maxPrediction = targetName;
                }

                probabilities.Add(targetName, probability);
            }

            return new ProbabilityPrediction(maxPrediction, probabilities);
        }

        /// <summary>
        /// Predicts the probabilities of a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var scores = Scores(observation);
            var probability = ScoresToProbability(scores);

            return probability;
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
        /// Predicts a set of probabilities
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
        /// Predict a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return PredictProbability(observation).Prediction;
        }

        /// <summary>
        /// Predicts a set of observations
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            return PredictProbability(observations).Select(p => p.Prediction)
                .ToArray();
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
        /// Loads a ClassificationGradientBoostModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static ClassificationGradientBoostModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<ClassificationGradientBoostModel>(reader);
        }

        /// <summary>
        /// Saves the ClassificationGradientBoostModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }
    }
}
