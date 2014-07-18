using SharpLearning.Containers.Views;
using SharpLearning.Metrics.Entropy;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpLearning.Containers;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    public sealed class BinarySplitSearcher : ISplitSearcher
    {
        readonly int m_minimumSplitSize;

        readonly List<Tuple<FeatureSplit, Tuple<Interval1D, Interval1D>>> m_workIntervals = 
            new List<Tuple<FeatureSplit, Tuple<Interval1D, Interval1D>>>();

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
           double[] weights, IEntropyMetric entropyMetric, Interval1D parentInterval, double parentEntropy)
        {
            AddThresholdIntervals(feature, targets, parentInterval);

            if (m_workIntervals.Count == 0)
            {
                return currentBestSplitResult;
            }

            int start = 0;
            int end = m_workIntervals.Count - 1;

            int middle = start + (end - start) / 2;
            var intervals = m_workIntervals[middle];
            
            var bestSplitResult = SplitResult(parentEntropy, intervals.Item1, intervals.Item2.Item1, intervals.Item2.Item2,
                parentInterval, weights, targets, featureIndex, entropyMetric);

            var bestInformationGain = 0.0;

            while ((end - start) >= 1)
            {
                var left = intervals.Item2.Item1;
                var leftIndex = (start + middle - 1) / 2;
                var leftIntervals = m_workIntervals[leftIndex];

                var leftSplit = SplitResult(parentEntropy, leftIntervals.Item1, leftIntervals.Item2.Item1, leftIntervals.Item2.Item2, 
                    parentInterval, weights, targets, featureIndex, entropyMetric);
                
                var right = intervals.Item2.Item2;
                var rightIndex = (end + 1 + middle) / 2;
                var rightIntervals = m_workIntervals[rightIndex];

                var rightSplit = SplitResult(parentEntropy, rightIntervals.Item1, rightIntervals.Item2.Item1, rightIntervals.Item2.Item2, 
                    parentInterval, weights, targets, featureIndex, entropyMetric);

                var leftDiff = leftSplit.BestInformationGain - bestInformationGain;
                var rightDiff = rightSplit.BestInformationGain - bestInformationGain;

                if (leftDiff <= 0 && rightDiff <= 0.0)
                {
                    return new FindSplitResult(bestSplitResult.BestSplitIndex, bestSplitResult.BestInformationGain, 
                        bestSplitResult.BestFeatureSplit, bestSplitResult.LeftIntervalEntropy, bestSplitResult.RightIntervalEntropy);
                }
                else if(leftSplit.BestInformationGain >= rightSplit.BestInformationGain)
                {
                    end = middle - 1;
                    middle = start + (end - start) / 2;
                    intervals = leftIntervals;
                    bestSplitResult = leftSplit;
                    bestInformationGain = bestSplitResult.BestInformationGain;
                }
                else
                {
                    start = middle + 1;
                    middle = start + (end - start) / 2;
                    intervals = rightIntervals;
                    bestSplitResult = rightSplit;
                    bestInformationGain = bestSplitResult.BestInformationGain;
                }
            }
            
            return new FindSplitResult(bestSplitResult.BestSplitIndex, bestSplitResult.BestInformationGain,
                bestSplitResult.BestFeatureSplit, bestSplitResult.LeftIntervalEntropy, bestSplitResult.RightIntervalEntropy);
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

                    if(leftSize >= 1 && rightSize >= 1) // interval needs to be large enough for a split
                    {
                        var leftInterval = Interval1D.Create(parentInterval.FromInclusive, j);
                        var rightInterval = Interval1D.Create(j, parentInterval.ToExclusive);

                        m_workIntervals.Add(new Tuple<FeatureSplit, Tuple<Interval1D, Interval1D>>(featureSplit,
                            new Tuple<Interval1D, Interval1D>(leftInterval, rightInterval)));
                    }
                }
            }
        }

        FindSplitResult SplitResult(double parentEntropy, FeatureSplit featureSplit, Interval1D leftInterval, Interval1D rightInterval,
            Interval1D parentInterval, double[] weights, double[] targets, int featureIndex, IEntropyMetric entropyMetric)
        {
            var leftEntropy = 0.0;
            var rightEntropy = 0.0;
            var informationGain = 0.0;

            if (weights.Length == 0)
            {
                leftEntropy = entropyMetric.Entropy(targets, leftInterval);
                rightEntropy = entropyMetric.Entropy(targets, rightInterval);

                var lengthInv = 1.0 / (parentInterval.Length);
                var leftRatio = leftInterval.Length * lengthInv;
                var rightRatio = rightInterval.Length * lengthInv;

                var wLeftEntropy = (leftRatio) * leftEntropy;
                var wRightEntropy = (rightRatio) * rightEntropy;

                informationGain = parentEntropy - (wLeftEntropy + wRightEntropy);
            }
            else
            {
                leftEntropy = entropyMetric.Entropy(targets, weights, leftInterval);
                rightEntropy = entropyMetric.Entropy(targets, weights, rightInterval);

                var parentWeight = weights.Sum(parentInterval);
                var leftWeight = weights.Sum(leftInterval);
                var rightWeight = weights.Sum(rightInterval);

                var lengthInv = 1.0 / (parentWeight);
                var leftRatio = leftWeight * lengthInv;
                var rightRatio = rightWeight * lengthInv;

                var wLeftEntropy = (leftRatio) * leftEntropy;
                var wRightEntropy = (rightRatio) * rightEntropy;

                informationGain = parentEntropy - (wLeftEntropy + wRightEntropy);
            }

            return new FindSplitResult(featureSplit.Index, informationGain,
                new FeatureSplit(featureSplit.Value, featureIndex), 
                new IntervalEntropy(leftInterval, leftEntropy), 
                new IntervalEntropy(rightInterval, rightEntropy));
        }
    }
}
