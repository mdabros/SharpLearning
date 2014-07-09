using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Entropy;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    /// <summary>
    /// A SplitSearcher seeks to find the most optimal split for the given feature and targets
    /// </summary>
    public interface ISplitSearcher
    {
        FindSplitResult FindBestSplit(FindSplitResult currentBestSplitResult, int featureIndex, double[] feature, double[] targets,
            IEntropyMetric entropyMetric, Interval1D parentInterval, double parentEntropy);
    }
}
