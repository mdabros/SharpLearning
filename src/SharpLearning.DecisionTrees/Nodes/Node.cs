using System;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Split node for binary decision tree
    /// </summary>
    [Serializable]
    public struct Node
    {
        /// <summary>
        /// Feature index used for split
        /// </summary>
        public readonly int FeatureIndex;

        /// <summary>
        /// Feature value used for split
        /// </summary>
        public readonly double Value;

        /// <summary>
        /// Right child tree index
        /// </summary>
        public readonly int RightIndex;
        
        /// <summary>
        /// Left child tree index
        /// </summary>
        public readonly int LeftIndex;

        /// <summary>
        /// Node tree index
        /// </summary>
        public readonly int NodeIndex;

        /// <summary>
        /// Probability tree index
        /// </summary>
        public readonly int LeafProbabilityIndex;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="featureIndex"></param>
        /// <param name="value"></param>
        /// <param name="leftIndex"></param>
        /// <param name="rightIndex"></param>
        /// <param name="nodeIndex"></param>
        /// <param name="leafProbabilityIndex"></param>
        public Node(int featureIndex, double value, int leftIndex, 
            int rightIndex, int nodeIndex, int leafProbabilityIndex)
        {
            FeatureIndex = featureIndex;
            Value = value;
            RightIndex = rightIndex;
            LeftIndex = leftIndex;
            NodeIndex = nodeIndex;
            LeafProbabilityIndex = leafProbabilityIndex;
        }

        /// <summary>
        /// Creates a default split node
        /// </summary>
        /// <returns></returns>
        public static Node Default()
        {
            return new Node(-1, 0.0, -1, -1, -1, -1);
        }
    }
}
