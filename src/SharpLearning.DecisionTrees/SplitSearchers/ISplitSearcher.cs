using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    /// <summary>
    /// A SplitSearcher seeks to find the most optimal split for the given feature and targets
    /// </summary>
    public interface ISplitSearcher
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="impurityCalculator"></param>
        /// <param name="feature"></param>
        /// <param name="targets"></param>
        /// <param name="parentInterval"></param>
        /// <param name="parentImpurity"></param>
        /// <returns></returns>
        SplitResult FindBestSplit(IImpurityCalculator impurityCalculator, double[] feature, double[] targets,
                   Interval1D parentInterval, double parentImpurity);
    }
}
