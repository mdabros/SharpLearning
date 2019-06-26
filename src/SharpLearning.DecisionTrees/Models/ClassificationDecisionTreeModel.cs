using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.DecisionTrees.Models
{
    /// <summary>
    /// Classification Decision tree model
    /// </summary>
    [Serializable]
    public sealed class ClassificationDecisionTreeModel : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly BinaryTree Tree;

        readonly double[] m_variableImportance;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tree"></param>
        public ClassificationDecisionTreeModel(BinaryTree tree)
        {
            Tree = tree ?? throw new ArgumentNullException(nameof(tree));
            m_variableImportance = tree.VariableImportance;
        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return Tree.Predict(observation);
        }

        /// <summary>
        /// Predicts a set of observations 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = Tree.Predict(observations.Row(i));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts the observation subset provided by indices
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations, int[] indices)
        {
            var rows = observations.RowCount;
            var predictions = new double[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                predictions[i] = Tree.Predict(observations.Row(indices[i]));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts a single observation with probabilities
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            return Tree.PredictProbability(observation);
        }

        /// <summary>
        /// Predicts a set of observations with probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = Tree.PredictProbability(observations.Row(i));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts the observation subset provided by indices with probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations, int[] indices)
        {
            var rows = observations.RowCount;
            var predictions = new ProbabilityPrediction[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                predictions[i] = Tree.PredictProbability(observations.Row(indices[i]));
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
            var max = m_variableImportance.Max();
            
            var scaledVariableImportance = m_variableImportance
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
        public double[] GetRawVariableImportance() => m_variableImportance;

        /// <summary>
        /// Loads a ClassificationDecisionTreeModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static ClassificationDecisionTreeModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<ClassificationDecisionTreeModel>(reader);
        }

        /// <summary>
        /// Saves the ClassificationDecisionTreeModel.
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
