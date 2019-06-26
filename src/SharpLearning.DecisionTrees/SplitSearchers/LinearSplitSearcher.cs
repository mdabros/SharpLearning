using System;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    /// <summary>
    /// Searches for the best split using a brute force approach. The searcher only considers splits 
    /// when both the threshold value and the target value has changed.  
    /// The implementation assumes that the features and targets have been sorted
    /// together using the features as sort criteria
    /// </summary>
    public sealed class LinearSplitSearcher : ISplitSearcher
    {
        readonly int m_minimumSplitSize;
        readonly double m_minimumLeafWeight;

        /// <summary>
        /// Searches for the best split using a brute force approach. The searcher only considers splits 
        /// when both the threshold value and the target value has changed.  
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        public LinearSplitSearcher(int minimumSplitSize)
            : this(minimumSplitSize, 0.0)
        {
        }

        /// <summary>
        /// Searches for the best split using a brute force approach. The searcher only considers splits 
        /// when both the threshold value and the target value has changed.  
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        /// <param name="minimumLeafWeight">Minimum leaf weight when splitting</param>
        public LinearSplitSearcher(int minimumSplitSize, double minimumLeafWeight)
        {
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            m_minimumSplitSize = minimumSplitSize;
            m_minimumLeafWeight = minimumLeafWeight;
        }

        /// <summary>
        /// Searches for the best split using a brute force approach. The searcher only considers splits 
        /// when both the threshold value and the target value has changed.
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="impurityCalculator"></param>
        /// <param name="feature"></param>
        /// <param name="targets"></param>
        /// <param name="parentInterval"></param>
        /// <param name="parentImpurity"></param>
        /// <returns></returns>
        public SplitResult FindBestSplit(IImpurityCalculator impurityCalculator, double[] feature, double[] targets, 
            Interval1D parentInterval, double parentImpurity)
        {

            var bestSplitIndex = -1;
            var bestThreshold = 0.0;
            var bestImpurityImprovement = 0.0;
            var bestImpurityLeft = 0.0;
            var bestImpurityRight = 0.0;
            
            int prevSplit = parentInterval.FromInclusive;
            var prevValue = feature[prevSplit];
            var prevTarget = targets[prevSplit];

            impurityCalculator.UpdateInterval(parentInterval);

            for (int j = prevSplit + 1; j < parentInterval.ToExclusive; j++)
            {
                var currentValue = feature[j];
                var currentTarget = targets[j];
                if (prevValue != currentValue && prevTarget != currentTarget)
                {
                    var currentSplit = j;
                    var leftSize = (double)(currentSplit - parentInterval.FromInclusive);
                    var rightSize = (double)(parentInterval.ToExclusive - currentSplit);

                    if (Math.Min(leftSize, rightSize) >= m_minimumSplitSize)
                    {
                        impurityCalculator.UpdateIndex(currentSplit);
                        
                        if (impurityCalculator.WeightedLeft < m_minimumLeafWeight ||
                            impurityCalculator.WeightedRight < m_minimumLeafWeight)
                        {
                            continue;
                        }

                        var improvement = impurityCalculator.ImpurityImprovement(parentImpurity);

                        if (improvement > bestImpurityImprovement)
                        {
                            var childImpurities = impurityCalculator.ChildImpurities(); // could be avoided

                            bestImpurityImprovement = improvement;
                            bestThreshold = (currentValue + prevValue) * 0.5;
                            bestSplitIndex = currentSplit;
                            bestImpurityLeft = childImpurities.Left;
                            bestImpurityRight = childImpurities.Right;
                        }

                        prevSplit = j;
                    }
                }

                prevValue = currentValue;
                prevTarget = currentTarget;
            }

            return new SplitResult(bestSplitIndex, bestThreshold,
                bestImpurityImprovement, bestImpurityLeft, bestImpurityRight);
        }
    }
}
