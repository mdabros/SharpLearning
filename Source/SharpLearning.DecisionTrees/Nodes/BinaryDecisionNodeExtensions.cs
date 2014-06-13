using System;

namespace SharpLearning.DecisionTrees.Nodes
{
    public static class BinaryDecisionNodeExtensions
    {
        /// <summary>
        /// Adds a child to the node. The position is determined by childPosition. 
        /// If the child position is Root the method does not add the child
        /// </summary>
        /// <param name="parent"></param>
        /// <param name="childPosition"></param>
        /// <param name="child"></param>
        public static void AddChild(this IBinaryDecisionNode parent, NodePositionType childPosition, IBinaryDecisionNode child)
        {
            switch (childPosition)
            {
                case NodePositionType.Root:
                    break;
                case NodePositionType.Left:
                    parent.Left = child;
                    break;
                case NodePositionType.Right:
                    parent.Right = child;
                    break;
                default:
                    throw new InvalidOperationException("Unsupported position type");
            }
        }
    }
}
