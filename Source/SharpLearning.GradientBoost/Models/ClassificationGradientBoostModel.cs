using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Serialization;
using SharpLearning.GradientBoost.GBMDecisionTree;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.GradientBoost.Models
{
    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public sealed class ClassificationGradientBoostModel : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly GBMTree[][] Trees;
        
        /// <summary>
        /// 
        /// </summary>
        public readonly double LearningRate;
        
        /// <summary>
        /// 
        /// </summary>
        public readonly double InitialLoss;
        
        /// <summary>
        /// 
        /// </summary>
        public readonly double[] TargetNames;
        
        /// <summary>
        /// 
        /// </summary>
        public readonly int FeatureCount;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trees"></param>
        /// <param name="targetNames"></param>
        /// <param name="learningRate"></param>
        /// <param name="initialLoss"></param>
        /// <param name="featureCount"></param>
        public ClassificationGradientBoostModel(GBMTree[][] trees, double[] targetNames, double learningRate, double initialLoss, int featureCount)
        {
            if (trees == null) { throw new ArgumentNullException("trees"); }
            if (targetNames == null) { throw new ArgumentException("targetNames"); }
            Trees = trees;
            LearningRate = learningRate;
            InitialLoss = initialLoss;
            TargetNames = targetNames;
            FeatureCount = featureCount;
        }

        /// <summary>
        /// Predicts a single observations using the combination of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            if(TargetNames.Length == 2)
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
            if (TargetNames.Length == 2)
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
            var rawVariableImportance = new double[FeatureCount];
            foreach (var treeIterations in Trees)
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
        public static ClassificationGradientBoostModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<ClassificationGradientBoostModel>(reader);
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
            var prediction = (probability >= 0.5) ? TargetNames[0] : TargetNames[1];
            return prediction;
        }

        double MultiClassPredict(double[] observation)
        {
            var probability = 0.0;
            var prediction = 0.0;

            for (int i = 0; i < TargetNames.Length; i++)
            {
                var currentProp = Probability(observation, i);
                if (currentProp > probability)
                {
                    prediction = TargetNames[i];
                    probability = currentProp;
                }
            }
            return prediction;
        }

        ProbabilityPrediction MultiClassProbabilityPredict(double[] observation)
        {
            var probabilities = new Dictionary<double, double>();
            for (int i = 0; i < TargetNames.Length; i++)
            {
                probabilities.Add(TargetNames[i], Probability(observation, i));
            }

            var prediction = probabilities.OrderBy(v => v.Value).Last().Key;
            return new ProbabilityPrediction(prediction, probabilities);
        }

        ProbabilityPrediction BinaryProbabilityPredict(double[] observation)
        {
            var probability = Probability(observation, 0);
            var prediction = (probability >= 0.5) ? TargetNames[0] : TargetNames[1];
            var probabilities = new Dictionary<double, double> { { TargetNames[1], 1.0 - probability }, { TargetNames[0], probability } };

            return new ProbabilityPrediction(prediction, probabilities);
        }

        double Probability(double[] observation, int targetIndex)
        {
            var iterations = Trees[targetIndex].Length;

            var prediction = InitialLoss;
            for (int i = 0; i < iterations; i++)
            {
                prediction += LearningRate * Trees[targetIndex][i].Predict(observation);
            }

            return Sigmoid(prediction);
        }

        double Sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }
    }
}
