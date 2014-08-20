using SharpLearning.Containers;
using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Binary tree 
    /// </summary>
    public sealed class BinaryTree
    {
        readonly List<Node> m_nodes;
        readonly List<double[]> m_probabilities;
        readonly double[] m_targetNames;
        public readonly double[] VariableImportance;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="nodes"></param>
        public BinaryTree(List<Node> nodes, List<double[]> probabilities, double[] targetNames, 
            double[] variableImportance)
        {
            if (nodes == null) { throw new ArgumentNullException("nodes"); }
            if (probabilities == null) { throw new ArgumentNullException("probabilities"); }
            if (targetNames == null) { throw new ArgumentNullException("targetNames"); }
            if (variableImportance == null) { throw new ArgumentNullException("variableImportance"); }
            m_nodes = nodes;
            m_probabilities = probabilities;
            m_targetNames = targetNames;
            VariableImportance = variableImportance;
        }

        /// <summary>
        /// Predicts using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return Predict(m_nodes[0], observation);
        }

        /// <summary>
        /// Predict probabilities using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            return PredictProbability(m_nodes[0], observation);
        }

        /// <summary>
        /// Predicts using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        double Predict(Node node, double[] observation)
        {
            if (node.FeatureIndex == -1.0)
            {
                return node.Value;
            }

            if (observation[node.FeatureIndex] <= node.Value)
            {
                return Predict(m_nodes[node.LeftIndex], observation);
            }
            else
            {
                return Predict(m_nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }

        /// <summary>
        /// Predict probabilities using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        ProbabilityPrediction PredictProbability(Node node, double[] observation)
        {
            if (node.FeatureIndex == -1.0)
            {
                var probabilities = m_probabilities[node.ProbabilityIndex];
                var targetProbabilities = new Dictionary<double, double>();

                for (int i = 0; i < m_targetNames.Length; i++)
                {
                    targetProbabilities.Add(m_targetNames[i], probabilities[i]);
                }

                return new ProbabilityPrediction(node.Value, targetProbabilities);
            }

            if (observation[node.FeatureIndex] <= node.Value)
            {
                return PredictProbability(m_nodes[node.LeftIndex], observation);
            }
            else
            {
                return PredictProbability(m_nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }
    }
}
