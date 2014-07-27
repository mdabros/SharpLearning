using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Nodes
{
    public enum NodePositionType { Root, Left, Right }

    /// <summary>
    /// Structure used for decision tree learning
    /// </summary>
    public struct DecisionNodeCreationItem
    {
        public readonly int ParentIndex;
        public readonly NodePositionType NodeType;
        public readonly Interval1D Interval;
        public readonly double Impurity;
        public readonly int NodeDepth;

        public DecisionNodeCreationItem(int parentIndex, NodePositionType nodeType, Interval1D interval, double impurity, int nodeDepth)
        {
            ParentIndex = parentIndex;
            NodeType = nodeType;
            Interval = interval;
            Impurity = impurity;
            NodeDepth = nodeDepth;
        }
    }
}
