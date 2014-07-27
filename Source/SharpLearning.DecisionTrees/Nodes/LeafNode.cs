using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Leaf node for binary decision tree
    /// </summary>
    public struct LeafNode : INode
    {
        public readonly int m_featureIndex;
        public readonly int m_nodeIndex;
        public readonly double m_value;
        public readonly Dictionary<double, double> m_probabilities;
        
        //Consider more memory efficient alternative to dictionary

        //public readonly double[] Targets;
        //public readonly double[] Probabilities;

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
        /// Probabilities if availible
        /// </summary>
        public Dictionary<double, double> Probabilities
        {
            get { return m_probabilities; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="featureIndex"></param>
        /// <param name="value"></param>
        /// <param name="nodeIndex"></param>
        /// <param name="probabilities"></param>
        public LeafNode(int featureIndex, double value, int nodeIndex, Dictionary<double, double> probabilities)
        {
            m_featureIndex = featureIndex;
            m_nodeIndex = nodeIndex;
            m_probabilities = probabilities;
            m_value = value;
        }
    }
}
