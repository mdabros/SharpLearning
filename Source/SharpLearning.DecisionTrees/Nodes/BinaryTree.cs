using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Binary tree 
    /// </summary>
    [Serializable]
    public sealed class BinaryTree
    {
        public readonly List<Node> Nodes;
        public readonly List<double[]> Probabilities;
        public readonly double[] TargetNames;
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
            Nodes = nodes;
            Probabilities = probabilities;
            TargetNames = targetNames;
            VariableImportance = variableImportance;
        }

        /// <summary>
        /// Predicts using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return Predict(Nodes[0], observation);
        }

        /// <summary>
        /// Predict probabilities using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            return PredictProbability(Nodes[0], observation);
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
                return Predict(Nodes[node.LeftIndex], observation);
            }
            else
            {
                return Predict(Nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }

        /// <summary>
        /// Returns the prediction node using a continous node strategy
        /// </summary>
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
        /// Predict probabilities using a continous node strategy
        /// </summary>
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

        /// <summary>
        /// Returns an array of index lists holding the index og which samples
        /// from observations goes to which leaf. The array uses the Node.ProbabilityIndex
        /// to index the leafs
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public List<int>[] LeafRegionIndices(F64Matrix observations, int[] indices)
        {
            var leafs = Probabilities.Count;
            var leafIndices = new List<int>[leafs];
            for (int i = 0; i < leafs; i++)
            {
                leafIndices[i] = new List<int>();
            }

            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                var node = PredictNode(Nodes[0], observations.GetRow(index));
                leafIndices[node.LeafProbabilityIndex].Add(index);
            }

            return leafIndices;
        }
    }
}
