using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    public static class NodeExtensions
    {
        /// <summary>
        /// Updates the parent node with the new child
        /// </summary>
        /// <param name="nodes"></param>
        /// <param name="parent"></param>
        /// <param name="child"></param>
        /// <param name="type"></param>
        public static void UpdateParent(this List<INode> nodes, INode parent, INode child, NodePositionType type)
        {
            switch (type)
            {
                case NodePositionType.Root:
                    break;
                case NodePositionType.Left:
                    nodes[parent.NodeIndex] = new SplitNode(parent.FeatureIndex, parent.Value,
                        child.NodeIndex, parent.RightIndex, parent.NodeIndex);
                    break;
                case NodePositionType.Right:
                    nodes[parent.NodeIndex] = new SplitNode(parent.FeatureIndex, parent.Value,
                        parent.LeftIndex, child.NodeIndex, parent.NodeIndex);
                    break;
                default:
                    throw new InvalidOperationException("Unsupported position type");
            }

        }
    }
}
