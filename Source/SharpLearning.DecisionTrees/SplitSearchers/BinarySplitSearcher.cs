using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    public sealed class BinarySplitSearcher : ISplitSearcher
    {
        readonly int m_minimumSplitSize;

        readonly List<FeatureSplit> m_workIntervals = new List<FeatureSplit>();

        /// <summary>
        /// Searches for the best split using a binary search strategy. The searcher only considers splits 
        /// when both the threshold value and the target value has changed.
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        public BinarySplitSearcher(int minimumSplitSize)
        {
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            m_minimumSplitSize = minimumSplitSize;
        }

        /// <summary>
        /// Searches for the best split using a binary search strategy. The searcher only considers splits 
        /// when both the threshold value and the target value has changed.
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="currentBestSplitResult"></param>
        /// <param name="feature"></param>
        /// <param name="targets"></param>
        /// <param name="parentInterval"></param>
        /// <param name="parentImpurity"></param>
        /// <param name="featureIndex"></param>
        /// <returns></returns>
        public SplitResult FindBestSplit(IImpurityCalculator impurityCalculator, double[] feature, double[] targets,
                   Interval1D parentInterval, double parentImpurity)
        {
            AddThresholdIntervals(feature, targets, parentInterval);
                        
            if (m_workIntervals.Count == 0)
            {
                return SplitResult.Initial();
            }

            var bestSplitIndex = -1;
            var bestThreshold = 0.0;
            var bestImpurityImprovement = 0.0;
            var bestImpurityLeft = 0.0;
            var bestImpurityRight = 0.0;

            int start = 0;
            int end = m_workIntervals.Count - 1;

            int middle = start + (end - start) / 2;
            var Splits = m_workIntervals[middle];

            while ((end - start) >= 1)
            {
                impurityCalculator.Reset();

                var leftIndex = (start + middle - 1) / 2;
                var leftSplit = m_workIntervals[leftIndex];
                impurityCalculator.UpdateIndex(leftSplit.Index);
                var leftImprovement = impurityCalculator.ImpurityImprovement(parentImpurity);
                                
                var rightIndex = (end + 1 + middle) / 2;
                var rightSplit = m_workIntervals[rightIndex];
                impurityCalculator.UpdateIndex(rightSplit.Index);
                var rightImprovement = impurityCalculator.ImpurityImprovement(parentImpurity);

                var leftDiff = leftImprovement - bestImpurityImprovement;
                var rightDiff = rightImprovement - bestImpurityImprovement;

                if (leftDiff <= 0 && rightDiff <= 0.0)
                {
                    return new SplitResult(bestSplitIndex, bestThreshold, bestImpurityImprovement,
                        bestImpurityLeft, bestImpurityRight);
                }
                else if (leftImprovement >= rightImprovement)
                {
                    end = middle - 1;
                    middle = start + (end - start) / 2;
    
                    bestSplitIndex = leftSplit.Index;
                    bestThreshold = leftSplit.Threshold;
                    bestImpurityImprovement = leftImprovement;

                    impurityCalculator.Reset();
                    impurityCalculator.UpdateIndex(leftSplit.Index);
                    var childImpurities = impurityCalculator.ChildImpurities();

                    bestImpurityLeft = childImpurities.Left;
                    bestImpurityRight = childImpurities.Right;
                }
                else
                {
                    start = middle + 1;
                    middle = start + (end - start) / 2;

                    bestSplitIndex = rightSplit.Index;
                    bestThreshold = rightSplit.Threshold;
                    bestImpurityImprovement = rightImprovement;

                    var childImpurities = impurityCalculator.ChildImpurities();

                    bestImpurityLeft = childImpurities.Left;
                    bestImpurityRight = childImpurities.Right;
                }
            }

            return new SplitResult(bestSplitIndex, bestThreshold, bestImpurityImprovement,
                bestImpurityLeft, bestImpurityRight);
        }

        void AddThresholdIntervals(double[] feature, double[] targets, Interval1D parentInterval)
        {
            m_workIntervals.Clear();

            for (int j = parentInterval.FromInclusive + 1; j < parentInterval.ToExclusive; j++)
            {
                // Add as candidate thresholds only adjacent values v[i] and v[i+1]
                // belonging to different classes, following the results by Fayyad
                // and Irani (1992). See footnote on Quinlan (1996).
                // Also only add unique values
                if (targets[j - 1] != targets[j] && feature[j - 1] != feature[j])
                {
                    var threshold = (feature[j - 1] + feature[j]) * 0.5;
                    var featureSplit = new FeatureSplit(threshold, j);

                    var leftSize = j - 1 - parentInterval.FromInclusive;
                    var rightSize = parentInterval.ToExclusive - j;

                    if (leftSize >= m_minimumSplitSize && rightSize >= m_minimumSplitSize)
                    {
                        m_workIntervals.Add(featureSplit);
                    }
                }
            }
        }
    }
}
