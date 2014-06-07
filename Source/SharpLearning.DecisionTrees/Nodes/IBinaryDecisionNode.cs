
namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Binary decision node
    /// </summary>
    public interface IBinaryDecisionNode
    {
        /// <summary>
        /// Feature index used for split
        /// </summary>
        int FeatureIndex { get; set; }

        /// <summary>
        /// Feature value used for split
        /// </summary>
        double Value { get; set; }

        /// <summary>
        /// Right child
        /// </summary>
        IBinaryDecisionNode Right { get; set; }

        /// <summary>
        /// Left child
        /// </summary>
        IBinaryDecisionNode Left { get; set; }

        /// <summary>
        /// Parent
        /// </summary>
        IBinaryDecisionNode Parent { get; set; }

        /// <summary>
        /// Is the node?
        /// </summary>
        /// <returns></returns>
        bool IsRoot();

        /// <summary>
        /// Is the node a leaf?
        /// </summary>
        /// <returns></returns>
        bool IsLeaf();

        /// <summary>
        /// Predicts using the decision tree structure
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        double Predict(double[] observation);
    }
}
