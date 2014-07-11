using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Entropy;
using System;
using System.Diagnostics;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    public sealed class BinarySplitSearcher : ISplitSearcher
    {
        readonly int m_minimumSplitSize;

        /// <summary>
        /// Searches for the best split using a binary search strategy. 
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
        /// Searches for the best split using a binary search strategy. 
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="currentBestSplitResult"></param>
        /// <param name="feature"></param>
        /// <param name="targets"></param>
        /// <param name="parentInterval"></param>
        /// <param name="parentEntropy"></param>
        /// <param name="featureIndex"></param>
        /// <returns></returns>
        public FindSplitResult FindBestSplit(FindSplitResult currentBestSplitResult, int featureIndex, double[] feature, double[] targets, 
            IEntropyMetric entropyMetric, Interval1D parentInterval, double parentEntropy)
        {
            var initialSplit = parentInterval.FromInclusive + parentInterval.ToExclusive / 2;
            var parentLeftSize = initialSplit - 1 - parentInterval.FromInclusive;
            var parentRighttSize = parentInterval.ToExclusive - initialSplit;

            if (parentLeftSize < 1 || parentRighttSize < 1)
            {
                return currentBestSplitResult;
            }

            var intialSplitResult = SplitResult(parentEntropy, parentInterval, initialSplit, 
                feature, targets, featureIndex, entropyMetric);
            
            var bestSplit = BinaryFindBestSplit(intialSplitResult, parentEntropy, parentInterval, 
                feature, targets, featureIndex, entropyMetric);

            if (bestSplit.BestInformationGain > currentBestSplitResult.BestInformationGain)
            {
                return new FindSplitResult(true, bestSplit.BestSplitIndex, bestSplit.BestInformationGain,
                    bestSplit.BestFeatureSplit, bestSplit.LeftIntervalEntropy, bestSplit.RightIntervalEntropy);
            }

            return bestSplit;
        }

        FindSplitResult BinaryFindBestSplit(FindSplitResult parentSplitResult, double parentEntropy, Interval1D parentInterval, 
            double[] feature, double[] targets, int featureIndex, IEntropyMetric entropyMetric)
        {
            var currentIndex = parentSplitResult.BestSplitIndex;
            var parentLeftSize = currentIndex - 1 - parentInterval.FromInclusive;
            var parentRighttSize = parentInterval.ToExclusive - currentIndex;

            if (parentLeftSize >= 1 && parentRighttSize >= 1)
            {
                var leftInterval = Interval1D.Create(parentInterval.FromInclusive, currentIndex - 1);
                var leftIndex = (parentInterval.FromInclusive + currentIndex - 1) / 2;

                var leftLeftSize = leftIndex - 1 - leftInterval.FromInclusive;
                var leftRightSize = leftInterval.ToExclusive - leftIndex;

                var rightInterval = Interval1D.Create(currentIndex, parentInterval.ToExclusive);
                var rightIndex = (currentIndex + 1 + parentInterval.ToExclusive) / 2;

                var rightLeftSize = rightIndex - 1 - rightInterval.FromInclusive;
                var rightRightSize = rightInterval.ToExclusive - rightIndex;

                if (leftLeftSize >= 1 && leftRightSize >= 1 && rightLeftSize >= 1 && rightRightSize >= 1)
                {
                    var leftSplitResult = SplitResult(parentEntropy, parentInterval, leftIndex,
                        feature, targets, featureIndex, entropyMetric);

                    var rightSplitResult = SplitResult(parentEntropy, parentInterval, rightIndex,
                        feature, targets, featureIndex, entropyMetric);

                    var leftRootDiff = leftSplitResult.BestInformationGain - parentSplitResult.BestInformationGain;
                    var rightRootDiff = rightSplitResult.BestInformationGain - parentSplitResult.BestInformationGain;

                    if (Math.Min(leftInterval.Length, rightInterval.Length) >= m_minimumSplitSize)
                    {
                        if (leftSplitResult.BestInformationGain > rightSplitResult.BestInformationGain &&
                           leftSplitResult.BestInformationGain > parentSplitResult.BestInformationGain)
                        {
                            return BinaryFindBestSplit(leftSplitResult, parentEntropy, parentInterval,
                                feature, targets, featureIndex, entropyMetric);
                        }
                        else if (rightSplitResult.BestInformationGain > leftSplitResult.BestInformationGain &&
                                 rightSplitResult.BestInformationGain > parentSplitResult.BestInformationGain)
                        {
                            return BinaryFindBestSplit(rightSplitResult, parentEntropy, parentInterval,
                                feature, targets, featureIndex, entropyMetric);
                        }
                    }
                }
            }

            return parentSplitResult;
        }

        FindSplitResult SplitResult(double parentEntropy, Interval1D parentInterval, int splitIndex, 
            double[] feature, double[] targets, int featureIndex, IEntropyMetric entropyMetric)
        {
            var leftInterval = Interval1D.Create(parentInterval.FromInclusive, splitIndex - 1);
            var rightInterval = Interval1D.Create(splitIndex, parentInterval.ToExclusive);

            var leftEntropy = entropyMetric.Entropy(targets, leftInterval);
            var rightEntropy = entropyMetric.Entropy(targets, rightInterval);
            var lengthInv = 1.0 / parentInterval.Length;
            var informationGain = parentEntropy - ((leftInterval.Length * lengthInv) * leftEntropy + (rightInterval.Length * lengthInv) * rightEntropy);

            return new FindSplitResult(false, splitIndex, informationGain,
                new FeatureSplit((feature[splitIndex] + feature[splitIndex - 1]) / 2.0, featureIndex), 
                new IntervalEntropy(leftInterval, leftEntropy), new IntervalEntropy(rightInterval, rightEntropy));
        }
    }
}
