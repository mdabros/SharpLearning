
namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Binary decision node
    /// </summary>
    public abstract class BinaryDicisionNodeBase : IBinaryDecisionNode
    {
        int m_featureIndex;
        double m_value;
        IBinaryDecisionNode m_right;
        IBinaryDecisionNode m_left;
        IBinaryDecisionNode m_parent;

        /// <summary>
        /// Feature index used for split
        /// </summary>
        public int FeatureIndex
        {
            get
            {
                return m_featureIndex;
            }
            set
            {
                m_featureIndex = value;
            }
        }

        /// <summary>
        /// Feature value used for split
        /// </summary>
        public double Value
        {
            get
            {
                return m_value;
            }
            set
            {
                m_value = value;
            }
        }

        /// <summary>
        /// Right child
        /// </summary>
        public IBinaryDecisionNode Right
        {
            get
            {
                return m_right;
            }
            set
            {
                m_right = value;
                m_right.Parent = null;
                m_right.Parent = this;

            }
        }

        /// <summary>
        /// Left child
        /// </summary>
        public IBinaryDecisionNode Left
        {
            get
            {
                return m_left;
            }
            set
            {
                m_left = value;
                m_left.Parent = null;
                m_left.Parent = this;
            }
        }

        /// <summary>
        /// Parent
        /// </summary>
        public IBinaryDecisionNode Parent
        {
            get
            {
                return m_parent;
            }
            set
            {
                m_parent = value;
            }
        }

        /// <summary>
        /// Is the node?
        /// </summary>
        /// <returns></returns>
        public bool IsRoot()
        {
            return m_parent == null;
        }

        /// <summary>
        /// Is the node a leaf?
        /// </summary>
        /// <returns></returns>
        public bool IsLeaf()
        {
            if (m_left != null) { return false; }
            if (m_right != null) { return false; }

            return true;
        }

        /// <summary>
        /// Predicts using the decision tree structure
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public abstract double Predict(double[] observation);
    }
}
