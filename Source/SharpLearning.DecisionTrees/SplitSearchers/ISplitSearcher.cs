using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Entropy;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    /// <summary>
    /// A SplitSearcher seeks to find the most optimal split for the given feature and targets
    /// </summary>
    public interface ISplitSearcher
    {
        /// <summary>
        /// Finds the most optimal split for a given feature and targets
        /// </summary>
        /// <param name="currentBestSplitResult"></param>
        /// <param name="featureIndex"></param>
        /// <param name="feature"></param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <param name="entropyMetric"></param>
        /// <param name="parentInterval"></param>
        /// <param name="parentEntropy"></param>
        /// <returns></returns>
        FindSplitResult FindBestSplit(FindSplitResult currentBestSplitResult, int featureIndex, double[] feature, double[] targets,
            double[] weights, IEntropyMetric entropyMetric, Interval1D parentInterval, double parentEntropy);
    }
}
