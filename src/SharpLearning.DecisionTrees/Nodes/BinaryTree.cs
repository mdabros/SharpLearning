using System;
using System.Collections.Generic;
using SharpLearning.Containers;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Binary tree 
    /// </summary>
    [Serializable]
    public sealed class BinaryTree
    {
        /// <summary>
        /// Tree Nodes
        /// </summary>
        public readonly List<Node> Nodes;
        
        /// <summary>
        /// Leaf node probabilities
        /// </summary>
        public readonly List<double[]> Probabilities;

        /// <summary>
        /// Target names
        /// </summary>
        public readonly double[] TargetNames;

        /// <summary>
        /// Raw variable importance
        /// </summary>
        public readonly double[] VariableImportance;

        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="nodes"></param>
        /// <param name="probabilities"></param>
        /// <param name="targetNames"></param>
        /// <param name="variableImportance"></param>
        public BinaryTree(List<Node> nodes, List<double[]> probabilities, double[] targetNames, 
            double[] variableImportance)
        {
            Nodes = nodes ?? throw new ArgumentNullException(nameof(nodes));
            Probabilities = probabilities ?? throw new ArgumentNullException(nameof(probabilities));
            TargetNames = targetNames ?? throw new ArgumentNullException(nameof(targetNames));
            VariableImportance = variableImportance ?? throw new ArgumentNullException(nameof(variableImportance));
        }

        /// <summary>
        /// Predicts using a continuous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return Predict(Nodes[0], observation);
        }

        /// <summary>
        /// Predict probabilities using a continuous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            return PredictProbability(Nodes[0], observation);
        }

        /// <summary>
        /// Predicts using a continuous node strategy
        /// </summary>
        /// <param name="node"></param>
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
                return Predict(Nodes[node.LeftIndex], observation);
            }
            else
            {
                return Predict(Nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }

        /// <summary>
        /// Returns the prediction node using a continuous node strategy
        /// </summary>
        /// <param name="node"></param>
        /// <param name="observation"></param>
        /// <returns></returns>
        Node PredictNode(Node node, double[] observation)
        {
            if (node.FeatureIndex == -1.0)
            {
                return node;
            }

            if (observation[node.FeatureIndex] <= node.Value)
            {
                return PredictNode(Nodes[node.LeftIndex], observation);
            }
            else
            {
                return PredictNode(Nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }

        /// <summary>
        /// Predict probabilities using a continuous node strategy
        /// </summary>
        /// <param name="node"></param>
        /// <param name="observation"></param>
        /// <returns></returns>
        ProbabilityPrediction PredictProbability(Node node, double[] observation)
        {
            if (node.FeatureIndex == -1.0)
            {
                var probabilities = Probabilities[node.LeafProbabilityIndex];
                var targetProbabilities = new Dictionary<double, double>();

                for (int i = 0; i < TargetNames.Length; i++)
                {
                    targetProbabilities.Add(TargetNames[i], probabilities[i]);
                }

                return new ProbabilityPrediction(node.Value, targetProbabilities);
            }

            if (observation[node.FeatureIndex] <= node.Value)
            {
                return PredictProbability(Nodes[node.LeftIndex], observation);
            }
            else
            {
                return PredictProbability(Nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }
    }
}
