using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Entropy;
using System;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    public sealed class LinearSplitSearcher : ISplitSearcher
    {
        readonly int m_minimumSplitSize;

        /// <summary>
        /// Searches for the best split using a brute force approach. 
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        public LinearSplitSearcher(int minimumSplitSize)
        {
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            m_minimumSplitSize = minimumSplitSize;
        }

        /// <summary>
        /// Searches for the best split using a brute force approach. 
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
            var newBestSplit = false;

            var bestSplitIndex = currentBestSplitResult.BestSplitIndex;
            var bestInformationGain = currentBestSplitResult.BestInformationGain;
            var bestLeftEntropy = currentBestSplitResult.LeftIntervalEntropy.Entropy;
            var bestRightEntropy = currentBestSplitResult.RightIntervalEntropy.Entropy;
            var bestLeftInterval = currentBestSplitResult.LeftIntervalEntropy.Interval;
            var bestRightInterval = currentBestSplitResult.RightIntervalEntropy.Interval;
            var bestFeatureSplit = currentBestSplitResult.BestFeatureSplit;

            int prevSplit = parentInterval.FromInclusive;
            var prevValue = feature[prevSplit];
            var prevTarget = targets[prevSplit];

            for (int j = prevSplit + 1; j < parentInterval.ToExclusive; j++)
            {
                var currentValue = feature[j];
                var currentTarget = targets[j];
                if (prevValue != currentValue && prevTarget != currentTarget)
                {
                    var currentSplit = j;
                    var leftSize = currentSplit - parentInterval.FromInclusive;
                    var rightSize = parentInterval.ToExclusive - currentSplit;

                    if (Math.Min(leftSize, rightSize) >= m_minimumSplitSize)
                    {
                        var leftInterval = Interval1D.Create(parentInterval.FromInclusive, currentSplit);
                        var rightInterval = Interval1D.Create(currentSplit, parentInterval.ToExclusive);

                        var leftEntropy = entropyMetric.Entropy(targets, leftInterval);
                        var rightEntropy = entropyMetric.Entropy(targets, rightInterval);
                        var lengthInv = 1.0 / parentInterval.Length;
                        var informationGain = parentEntropy - ((leftSize * lengthInv) * leftEntropy + (rightSize * lengthInv) * rightEntropy);

                        if (informationGain > bestInformationGain)
                        {
                            bestSplitIndex = currentSplit;
                            bestFeatureSplit = new FeatureSplit((currentValue + prevValue) * 0.5, featureIndex);
                            bestInformationGain = informationGain;
                            bestLeftInterval = leftInterval;
                            bestRightInterval = rightInterval;
                            bestLeftEntropy = leftEntropy;
                            bestRightEntropy = rightEntropy;
                            newBestSplit = true;
                        }

                        prevSplit = j;
                    }
                }

                prevValue = currentValue;
                prevTarget = currentTarget;
            }

            return new FindSplitResult(newBestSplit, bestSplitIndex, bestInformationGain, bestFeatureSplit, 
                new IntervalEntropy(bestLeftInterval, bestLeftEntropy),
                new IntervalEntropy(bestRightInterval, bestRightEntropy));
        }
    }
}
