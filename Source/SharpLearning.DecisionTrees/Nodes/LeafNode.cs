using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Leaf node for binary decision tree
    /// </summary>
    public struct LeafNode : INode
    {
        readonly int m_featureIndex;
        readonly int m_nodeIndex;
        readonly double m_value;
        readonly double[] m_targetNames;
        readonly double[] m_probabilities;

        /// <summary>
        /// Feature index used for split
        /// </summary>
        public int FeatureIndex
        {
            get { return m_featureIndex; }
        }

        /// <summary>
        /// Feature value used for split
        /// </summary>
        public double Value
        {
            get { return m_value; }
        }

        /// <summary>
        /// Leaf does not support right index
        /// </summary>
        public int RightIndex
        {
            get { throw new NotSupportedException("Leaf nodes does not support RightIndex"); }
        }

        /// <summary>
        /// Leaf does not support left index
        /// </summary>
        public int LeftIndex
        {
            get { throw new NotSupportedException("Leaf nodes does not support LeftIndex"); }
        }

        /// <summary>
        /// Node tree index
        /// </summary>
        public int NodeIndex
        {
            get { return m_nodeIndex; }
        }

        /// <summary>
        /// The probability estimates. 
        /// Order is same as TargetNames
        /// </summary>
        public double[] Probabilities
        {
            get { return m_probabilities; }
        }

        /// <summary>
        /// The availible target names
        /// </summary>
        public double[] TargetNames
        {
            get { return m_targetNames; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="featureIndex"></param>
        /// <param name="value"></param>
        /// <param name="nodeIndex"></param>
        /// <param name="probabilities"></param>
        public LeafNode(int featureIndex, double value, int nodeIndex, 
            double[] targetNames, double[] probabilities)
        {
            m_featureIndex = featureIndex;
            m_value = value;
            m_nodeIndex = nodeIndex;
            m_targetNames = targetNames;
            m_probabilities = probabilities;

        }
    }
}
