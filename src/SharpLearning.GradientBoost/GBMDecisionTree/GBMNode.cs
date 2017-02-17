using System;

namespace SharpLearning.GradientBoost.GBMDecisionTree
{

    /// <summary>
    /// Decision tree node for Gradient boost decision tree
    /// </summary>
    [Serializable]
    public class GBMNode
    {
        /// <summary>
        /// Index of the feature that the node splits on
        /// </summary>
        public int FeatureIndex;

        /// <summary>
        /// Value of the feature that the node splits on
        /// </summary>
        public double SplitValue;

        /// <summary>
        /// The error on the left side of the split
        /// </summary>
        public double LeftError;

        /// <summary>
        /// The error on the right side of the split
        /// </summary>
        public double RightError;

        /// <summary>
        /// Left constant (fitted value) of the split
        /// </summary>
        public double LeftConstant;
        
        /// <summary>
        /// Right constant (fitted value) of the split
        /// </summary>
        public double RightConstant;
        
        /// <summary>
        /// Depth of the node in the decision tree
        /// </summary>
        public int Depth;

        /// <summary>
        /// Index of the left child node the node in the decision tree array
        /// </summary>
        public int LeftIndex = -1;
 
        /// <summary>
        /// Index of the left child node the node in the decision tree array
        /// </summary>
        public int RightIndex = -1;
        
        /// <summary>
        /// The number of observations in the node
        /// </summary>
        public int SampleCount;
    }
}
