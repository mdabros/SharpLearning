using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Structure used for decision tree learning
    /// </summary>
    public struct DecisionNodeCreationItem
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly int ParentIndex;

        /// <summary>
        /// 
        /// </summary>
        public readonly NodePositionType NodeType;

        /// <summary>
        /// 
        /// </summary>
        public readonly Interval1D Interval;

        /// <summary>
        /// 
        /// </summary>
        public readonly double Impurity;

        /// <summary>
        /// 
        /// </summary>
        public readonly int NodeDepth;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="parentIndex"></param>
        /// <param name="nodeType"></param>
        /// <param name="interval"></param>
        /// <param name="impurity"></param>
        /// <param name="nodeDepth"></param>
        public DecisionNodeCreationItem(int parentIndex, NodePositionType nodeType, Interval1D interval, 
            double impurity, int nodeDepth)
        {
            ParentIndex = parentIndex;
            NodeType = nodeType;
            Interval = interval;
            Impurity = impurity;
            NodeDepth = nodeDepth;
        }
    }
}
