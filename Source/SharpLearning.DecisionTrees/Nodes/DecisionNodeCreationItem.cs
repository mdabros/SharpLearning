using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Nodes
{
    public enum NodePositionType { Root, Left, Right }

    /// <summary>
    /// Structure used for decision tree learning
    /// </summary>
    public struct DecisionNodeCreationItem
    {
        public readonly IBinaryDecisionNode Parent;
        public readonly NodePositionType NodeType;
        public readonly Interval1D Interval;
        public readonly double Entropy;
        public readonly int NodeDepth;

        public DecisionNodeCreationItem(IBinaryDecisionNode node, NodePositionType nodeType, Interval1D interval, double entropy, int nodeDepth)
        {
            Parent = node;
            NodeType = nodeType;
            Interval = interval;
            Entropy = entropy;
            NodeDepth = nodeDepth;
        }
    }
}
