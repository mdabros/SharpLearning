using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Extension methods for node
    /// </summary>
    public static class NodeExtensions
    {
        /// <summary>
        /// Updates the parent node with the new child
        /// </summary>
        /// <param name="nodes"></param>
        /// <param name="parent"></param>
        /// <param name="child"></param>
        /// <param name="type"></param>
        public static void UpdateParent(this List<Node> nodes, Node parent, Node child, NodePositionType type)
        {
            switch (type)
            {
                case NodePositionType.Root:
                    break;
                case NodePositionType.Left:
                    nodes[parent.NodeIndex] = new Node(parent.FeatureIndex, parent.Value,
                        child.NodeIndex, parent.RightIndex, parent.NodeIndex, parent.LeafProbabilityIndex);
                    break;
                case NodePositionType.Right:
                    nodes[parent.NodeIndex] = new Node(parent.FeatureIndex, parent.Value,
                        parent.LeftIndex, child.NodeIndex, parent.NodeIndex, parent.LeafProbabilityIndex);
                    break;
                default:
                    throw new InvalidOperationException("Unsupported position type");
            }

        }
    }
}
