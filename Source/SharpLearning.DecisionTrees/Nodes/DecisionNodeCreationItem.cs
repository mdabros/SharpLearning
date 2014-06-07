using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Nodes
{
    public enum NodeType { Root, Left, Right }

    /// <summary>
    /// Structure used for decision tree learning
    /// </summary>
    public struct DecisionNodeCreationItem
    {
        public readonly IBinaryDecisionNode Parent;
        public readonly NodeType NodeType;
        public readonly Interval1D Interval;
        public readonly double Entropy;

        public DecisionNodeCreationItem(IBinaryDecisionNode node, NodeType nodeType, Interval1D interval, double entropy)
        {
            Parent = node;
            NodeType = nodeType;
            Interval = interval;
            Entropy = entropy;
        }
    }
}
