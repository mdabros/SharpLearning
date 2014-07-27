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
        readonly List<INode> m_nodes;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="nodes"></param>
        public BinaryTree(List<INode> nodes)
        {
            if (nodes == null) { throw new ArgumentNullException("nodes"); }
            m_nodes = nodes;
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
        double Predict(INode node, double[] observation)
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
        ProbabilityPrediction PredictProbability(INode node, double[] observation)
        {
            if (node.FeatureIndex == -1.0)
            {
                return new ProbabilityPrediction(node.Value, node.Probabilities);
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
