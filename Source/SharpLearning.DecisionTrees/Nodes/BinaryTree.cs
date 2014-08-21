using SharpLearning.Containers;
using SharpLearning.Containers.Views;
using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Binary tree 
    /// </summary>
    public sealed class BinaryTree
    {
        public readonly List<Node> Nodes;
        public readonly List<Interval1D> LeafIntervals;
        public readonly List<double[]> Probabilities;
        public readonly double[] TargetNames;
        public readonly double[] VariableImportance;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="nodes"></param>
        public BinaryTree(List<Node> nodes, List<double[]> probabilities, List<Interval1D> leafIntervals,
            double[] targetNames, double[] variableImportance)
        {
            if (nodes == null) { throw new ArgumentNullException("nodes"); }
            if (probabilities == null) { throw new ArgumentNullException("probabilities"); }
            if (leafIntervals == null) { throw new ArgumentNullException("leafIntervals"); }
            if (targetNames == null) { throw new ArgumentNullException("targetNames"); }
            if (variableImportance == null) { throw new ArgumentNullException("variableImportance"); }
            Nodes = nodes;
            Probabilities = probabilities;
            LeafIntervals = leafIntervals;
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
    }
}
