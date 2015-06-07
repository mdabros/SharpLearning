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
        readonly List<GBMNode> m_nodes;

        public GBMTree(List<GBMNode> nodes)
        {
            if (nodes == null) { throw new ArgumentNullException("nodes"); }
            m_nodes = nodes;
        }

        /// <summary>
        /// Predicts a series of observations
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
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            if (m_nodes.Count == 1) //only root
            {
                return m_nodes[0].LeftConstant; // left right are equal
            }

            var leaf = Predict(m_nodes[1], 1, observation);

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

            var node = m_nodes[nodeIndex];

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
        // Variable importances are based on the work each variable does (error reduction).
        // the scores at each split are scaled by the amount of data the node splits
        // if a node splits on 30% of the total data it will add
        // errorReduction * 0.3 to its importance score.
        // Based on this explanation:
        // http://www.salford-systems.com/videos/tutorials/how-to/variable-importance-in-cart
        /// </summary>
        /// <param name="rawVariableImportances"></param>
        public void AddRawVariableImportances(double[] rawVariableImportances)
        {
            if(m_nodes.Count == 1) { return; } // no splits no importance

            var rootError = m_nodes[0].LeftError;
            var totalSampleCount = m_nodes[0].SampleCount;
            AddRecursive(rawVariableImportances, m_nodes[1], rootError, totalSampleCount);
        }

        void AddRecursive(double[] rawFeatureImportances, GBMNode node, double previousError, int totalSampleCount)
        {
            var error = node.LeftError + node.RightError;
            var reduction = previousError - error;
            rawFeatureImportances[node.FeatureIndex] += reduction * reduction *
                (double)node.SampleCount / (double)totalSampleCount;
            
            if(node.LeftIndex != -1)
            {
                AddRecursive(rawFeatureImportances, m_nodes[node.LeftIndex], error, totalSampleCount);
            }

            if (node.RightIndex != -1)
            {
                AddRecursive(rawFeatureImportances, m_nodes[node.RightIndex], error, totalSampleCount);
            }
        }

        /// <summary>
        /// Traces the nodes in indexed order
        /// </summary>
        public void TraceNodesIndexed()
        {
            for (int i = 0; i < m_nodes.Count; i++)
            {
                var node = m_nodes[i];
                System.Diagnostics.Trace.WriteLine("Index: " + i + " SplitValue: " + m_nodes[i].SplitValue +
                    " left: " + node.LeftConstant + " right: " + node.RightConstant);
            }
        }

        /// <summary>
        /// Traces the nodes sorted by depth
        /// </summary>
        public void TraceNodesDepth()
        {
            var depths = m_nodes.Select(n => n.Depth)
                .OrderBy(d => d).Distinct();

            foreach (var depth in depths)
            {
                var index = 0;
                var nodes = m_nodes.Select(n => new { Node = n, Index = index++ }).Where(n => n.Node.Depth == depth);
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
