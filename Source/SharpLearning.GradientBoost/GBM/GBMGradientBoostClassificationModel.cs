using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Serialization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public sealed class GBMGradientBoostClassificationModel : IPredictor<double>, IPredictor<ProbabilityPrediction>
    {
        readonly GBMTree[][] m_trees;
        readonly double m_learningRate;
        readonly double m_initialLoss;
        readonly double[] m_targetNames;
        readonly int m_featureCount;

        public GBMGradientBoostClassificationModel(GBMTree[][] trees, double[] targetNames, double learningRate, double initialLoss, int featureCount)
        {
            if (trees == null) { throw new ArgumentNullException("trees"); }
            if (targetNames == null) { throw new ArgumentException("targetNames"); }
            m_trees = trees;
            m_learningRate = learningRate;
            m_initialLoss = initialLoss;
            m_targetNames = targetNames;
            m_featureCount = featureCount;
        }

        /// <summary>
        /// Predicts a single observations using the combination of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            if(m_targetNames.Length == 2)
            {
                return BinaryPredict(observation);
            }
            else
            {
                return MultiClassPredict(observation);
            }
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
        /// Predicts a single observation with probabilities
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            if (m_targetNames.Length == 2)
            {
                return BinaryProbabilityPredict(observation);
            }
            else
            {
                return MultiClassProbabilityPredict(observation);
            }
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
            var rawVariableImportance = GetRawVariableImportance();
            var max = rawVariableImportance.Max();

            var scaledVariableImportance = rawVariableImportance
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
            foreach (var treeIterations in m_trees)
            {
                foreach (var tree in treeIterations)
                {
                    tree.AddRawVariableImportances(rawVariableImportance);
                }
            }

            return rawVariableImportance;
        }

        /// <summary>
        /// Loads a GBMGradientBoostClassificationModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static GBMGradientBoostClassificationModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<GBMGradientBoostClassificationModel>(reader);
        }

        /// <summary>
        /// Saves the GBMGradientBoostClassificationModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }


        double BinaryPredict(double[] observation)
        {
            var probability = Probability(observation, 0);
            var prediction = (probability >= 0.5) ? m_targetNames[0] : m_targetNames[1];
            return prediction;
        }

        double MultiClassPredict(double[] observation)
        {
            var probability = 0.0;
            var prediction = 0.0;

            for (int i = 0; i < m_targetNames.Length; i++)
            {
                var currentProp = Probability(observation, i);
                if (currentProp > probability)
                {
                    prediction = m_targetNames[i];
                    probability = currentProp;
                }
            }
            return prediction;
        }

        ProbabilityPrediction MultiClassProbabilityPredict(double[] observation)
        {
            var probabilities = new Dictionary<double, double>();
            for (int i = 0; i < m_targetNames.Length; i++)
            {
                probabilities.Add(m_targetNames[i], Probability(observation, i));
            }

            var prediction = probabilities.OrderBy(v => v.Value).Last().Key;
            return new ProbabilityPrediction(prediction, probabilities);
        }

        ProbabilityPrediction BinaryProbabilityPredict(double[] observation)
        {
            var probability = Probability(observation, 0);
            var prediction = (probability >= 0.5) ? m_targetNames[0] : m_targetNames[1];
            var probabilities = new Dictionary<double, double> { { m_targetNames[1], 1.0 - probability }, { m_targetNames[0], probability } };

            return new ProbabilityPrediction(prediction, probabilities);
        }

        double Probability(double[] observation, int targetIndex)
        {
            var iterations = m_trees[targetIndex].Length;

            var prediction = m_initialLoss;
            for (int i = 0; i < iterations; i++)
            {
                prediction += m_learningRate * m_trees[targetIndex][i].Predict(observation);
            }

            return Sigmoid(prediction);
        }

        double Sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }
    }
}
