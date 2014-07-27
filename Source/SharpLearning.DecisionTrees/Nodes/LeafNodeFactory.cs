
namespace SharpLearning.DecisionTrees.Nodes
{
    /// <summary>
    /// Creates leaf nodes based on the input parameters
    /// </summary>
    public sealed class LeafNodeFactory
    {
        public INode Create(int featureIndex, double value, int nodeIndex,
            double[] targetNames, double[] probabilities)
        {
            if(targetNames.Length == 0 || probabilities.Length == 0)
            {
                return new LeafNode(featureIndex, value, nodeIndex);
            }
            else
            {
                return new ProbabilityLeafNode(featureIndex, value, nodeIndex,
                    targetNames, probabilities);
            }
        }
    }
}
