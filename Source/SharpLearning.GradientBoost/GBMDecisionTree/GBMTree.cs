using SharpLearning.Containers.Matrices;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.GradientBoost.GBMDecisionTree
{
    /// <summary>
    /// Binary decision tree based on GBMNodes.
    /// </summary>
    [Serializable]
    public class GBMTree
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly List<GBMNode> Nodes;
                
        /// <summary>
        /// Creates a GBMTree from the provided nodes
        /// </summary>
        /// <param name="nodes"></param>
        public GBMTree(List<GBMNode> nodes)
        {
            if (nodes == null) { throw new ArgumentNullException("nodes"); }
            Nodes = nodes;
        }

        /// <summary>
        /// Predicts a series of observations
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new double[rows];

            Predict(observations, predictions);

            return predictions;
        }

        /// <summary>
        /// Predicts a series of observations.
        /// can reuse predictions array if several predictions are made.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="predictions"></param>
        public void Predict(F64Matrix observations, double[] predictions)
        {
            var rows = observations.RowCount;
            var features = new double[observations.ColumnCount];
            for (int i = 0; i < rows; i++)
            {
                observations.Row(i, features);
                predictions[i] = Predict(features);
            }
        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            if (Nodes.Count == 1) //only root
            {
                return Nodes[0].LeftConstant; // left right are equal
            }

            var leaf = Predict(Nodes[1], 1, observation);

            if (observation[leaf.FeatureIndex] < leaf.SplitValue)
            {
                return leaf.LeftConstant;
            }
            else
            {
                return leaf.RightConstant;
            }
        }

        GBMNode Predict(GBMNode parent, int nodeIndex, double[] observation)
        {
            if (nodeIndex == -1)
            {
                return parent;
            }

            var node = Nodes[nodeIndex];

            if (observation[node.FeatureIndex] < node.SplitValue)
            {
                return Predict(node, node.LeftIndex, observation); // left
            }
            else
            {
                return Predict(node, node.RightIndex, observation); // right
            }
        }

        /// <summary>
        /// Variable importances are based on the work each variable does (error reduction).
        /// the scores at each split are scaled by the amount of data the node splits
        /// if a node splits on 30% of the total data it will add
        /// errorReduction * 0.3 to its importance score.
        /// Based on this explanation:
        /// http://www.salford-systems.com/videos/tutorials/how-to/variable-importance-in-cart
        /// </summary>
        /// <param name="rawVariableImportances"></param>
        public void AddRawVariableImportances(double[] rawVariableImportances)
        {
            if(Nodes.Count == 1) { return; } // no splits no importance

            var rootError = Nodes[0].LeftError;
            var totalSampleCount = Nodes[0].SampleCount;
            AddRecursive(rawVariableImportances, Nodes[1], rootError, totalSampleCount);
        }

        void AddRecursive(double[] rawFeatureImportances, GBMNode node, double previousError, int totalSampleCount)
        {
            var error = node.LeftError + node.RightError;
            var reduction = previousError - error;
            rawFeatureImportances[node.FeatureIndex] += reduction * reduction *
                (double)node.SampleCount / (double)totalSampleCount;
            
            if(node.LeftIndex != -1)
            {
                AddRecursive(rawFeatureImportances, Nodes[node.LeftIndex], error, totalSampleCount);
            }

            if (node.RightIndex != -1)
            {
                AddRecursive(rawFeatureImportances, Nodes[node.RightIndex], error, totalSampleCount);
            }
        }

        /// <summary>
        /// Traces the nodes in indexed order
        /// </summary>
        public void TraceNodesIndexed()
        {
            for (int i = 0; i < Nodes.Count; i++)
            {
                var node = Nodes[i];
                System.Diagnostics.Trace.WriteLine("Index: " + i + " SplitValue: " + Nodes[i].SplitValue +
                    " left: " + node.LeftConstant + " right: " + node.RightConstant);
            }
        }

        /// <summary>
        /// Traces the nodes sorted by depth
        /// </summary>
        public void TraceNodesDepth()
        {
            var depths = Nodes.Select(n => n.Depth)
                .OrderBy(d => d).Distinct();

            foreach (var depth in depths)
            {
                var index = 0;
                var nodes = Nodes.Select(n => new { Node = n, Index = index++ }).Where(n => n.Node.Depth == depth);
                var text = string.Empty;

                foreach (var node in nodes)
                {
                    text += "(";
                    text += string.Format("{0:0.000} I:{1} ",
                        node.Node.SplitValue, node.Node.FeatureIndex);

                    if(node.Node.LeftIndex == -1)
                    {
                        text += string.Format("L: {0:0.000} ", node.Node.LeftConstant);
                    }

                    if (node.Node.RightIndex == -1)
                    {
                        text += string.Format("R: {0:0.000} ", node.Node.RightConstant);
                    }

                    text += ")";
                }
                Trace.WriteLine(text);
            }
        }
    }
}
