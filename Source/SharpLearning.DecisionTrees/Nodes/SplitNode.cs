using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Split node for binary decision tree
    /// </summary>
    public struct SplitNode : INode
    {
        readonly int m_featureIndex;
        readonly double m_value;

        readonly int m_rightIndex;
        readonly int m_leftIndex;

        readonly int m_nodeIndex;

        /// <summary>
        /// Feature index used for split
        /// </summary>
        public int FeatureIndex 
        { get { return m_featureIndex; } }

        /// <summary>
        /// Feature value used for split
        /// </summary>
        public double Value 
        { get { return m_value; } }

        /// <summary>
        /// Right child tree index
        /// </summary>
        public int RightIndex 
        { get { return m_rightIndex; } }

        /// <summary>
        /// Left child tree index
        /// </summary>
        public int LeftIndex 
        { get { return m_leftIndex; } }

        /// <summary>
        /// Node tree index
        /// </summary>
        public int NodeIndex 
        { get { return m_nodeIndex; } }

        /// <summary>
        /// Probabilities are not supported in split node
        /// </summary>
        public Dictionary<double, double> Probabilities
        { get { throw new NotSupportedException("Split nodes does not support probabilities"); } }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="featureIndex"></param>
        /// <param name="value"></param>
        /// <param name="leftIndex"></param>
        /// <param name="rightIndex"></param>
        /// <param name="nodeIndex"></param>
        public SplitNode(int featureIndex, double value, int leftIndex, int rightIndex, int nodeIndex)
        {
            m_featureIndex = featureIndex;
            m_value = value;
            m_rightIndex = rightIndex;
            m_leftIndex = leftIndex;
            m_nodeIndex = nodeIndex;
        }

        /// <summary>
        /// Creates a default split node
        /// </summary>
        /// <returns></returns>
        public static SplitNode Default()
        {
            return new SplitNode(-1, 0.0, -1, -1, -1);
        }
    }
}
