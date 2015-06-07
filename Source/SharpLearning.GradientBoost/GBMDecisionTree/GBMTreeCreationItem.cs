
namespace SharpLearning.GradientBoost.GBMDecisionTree
{
    /// <summary>
    /// Tree creation item for learning GradientBoost regression tree
    /// </summary>
    public class GBMTreeCreationItem
    {
        /// <summary>
        /// Information about current split
        /// </summary>
        public GBMSplitInfo Values;

        /// <summary>
        /// Current observations in the sample
        /// </summary>
        public bool[] InSample;

        /// <summary>
        /// Current depth
        /// </summary>
        public int Depth;

        /// <summary>
        /// Parent node of the split
        /// </summary>
        public GBMNode Parent;
    }
}
