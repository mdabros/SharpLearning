using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System;
using System.Collections.Generic;

namespace SharpLearning.GradientBoost.GBM
{
    public sealed class GBMRegressionTreeLearner
    {
        /// <summary>
        // x the dataset
        // y the values we t the tree on
        // orderedElements the ordered indexes according to each feature
        // inSample a binary vector to indicate if x[i] is in the subsample
        
        // P the number of features
        // N the number of examples in the subsample
        // S the total sum
        // S2 the total sum of squares
        // depth the remaining depth levels
        // result the resulting tree
        // k the node number
        /// </summary>

        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;
        readonly int m_maximumTreeDepth;

        /// <summary>
        /// Fites a regression decision tree using a set presorted indices for each feature.
        /// </summary>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        public GBMRegressionTreeLearner(int maximumTreeDepth = 2000, int minimumSplitSize = 1, double minimumInformationGain = 1E-6)
        {
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            m_maximumTreeDepth = maximumTreeDepth;
            m_minimumSplitSize = minimumSplitSize;
            m_minimumInformationGain = minimumInformationGain;
        }

        /// <summary>
        /// Fites a regression decision tree using a set presorted indices for each feature.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="orderedElements">jagged array of sorted indices corresponding to each features</param>
        /// <param name="inSample">bool array containing the samples to use</param>
        /// <param name="s">sum</param>
        /// <param name="s2">sum of squares</param>
        /// <param name="n">number of samples</param>
        /// <returns></returns>
        public GBMTree Learn(F64Matrix observations, double[] targets, int[][] orderedElements, bool[] inSample, 
            double s, double s2, int n)
        {

            var bestLeft = GBMSplitInfo.NewEmpty();
            var bestRight = GBMSplitInfo.NewEmpty();

            var rootBestConstant = s / (double)n;
            var rootCost = s2 - s * s / (double)n;
            var root = new GBMNode() { FeatureIndex = -1, SplitValue = -1, LeftError = rootCost, RightError = rootCost, 
                LeftConstant = rootBestConstant, RightConstant = rootBestConstant };
            
            var nodes = new List<GBMNode> { root };

            var rootValues = new GBMSplitInfo { Samples = n, Sum = s, SumOfSquares = s2, 
                Cost = rootCost, BestConstant = rootBestConstant  };

            var stack = new Queue<GBMTreeCreationItem>(100);
            stack.Enqueue(new GBMTreeCreationItem { Values = rootValues, InSample = inSample, Depth = 1 });

            var featureCount = observations.GetNumberOfColumns();
            var nodeIndex = 0;

            while (stack.Count > 0)
            {
                var parentItem = stack.Dequeue();
                var parentInSample = parentItem.InSample;

                var isLeaf = (parentItem.Depth >= m_maximumTreeDepth);

                var bestSplit = new GBMSplit
                {
                    Depth = parentItem.Depth,
                    FeatureIndex = -1,
                    SplitIndex = -1,
                    SplitValue = -1,
                    Cost = double.MaxValue,
                    LeftConstant = -1,
                    RightConstant = -1
                    
                };

                for (int i = 0; i < featureCount; i++)
                {
                    var left = GBMSplitInfo.NewEmpty();
                    var right = new GBMSplitInfo
                    {
                        Samples = parentItem.Values.Samples,
                        Sum = parentItem.Values.Sum,
                        SumOfSquares = parentItem.Values.SumOfSquares,
                        Cost = parentItem.Values.Cost,
                        BestConstant = parentItem.Values.BestConstant
                    };

                    var orderedIndices = orderedElements[i];

                    for (int j = 0; j < orderedIndices.Length - 1; j++)
                    {
                        var index = orderedIndices[j];

                        if (parentInSample[index])
                        {

                            var y2 = targets[index] * targets[index];

                            left.Samples++;
                            left.Sum += targets[index];
                            left.SumOfSquares += y2;
                            left.Cost = left.SumOfSquares - (left.Sum * left.Sum / (double)left.Samples);
                            left.BestConstant = left.Sum / (double)left.Samples;

                            right.Samples--;
                            right.Sum -= targets[index];
                            right.SumOfSquares -= y2;
                            right.Cost = right.SumOfSquares - (right.Sum * right.Sum / (double)right.Samples);
                            right.BestConstant = right.Sum / (double)right.Samples;

                            if (Math.Min(left.Samples, right.Samples) >= m_minimumSplitSize)
                            {
                                var nextIndex = NextIndexInSample(targets, parentInSample, orderedIndices, j);

                                if (observations.GetItemAt(index, i) != observations.GetItemAt(nextIndex, i))
                                {
                                    var cost = left.Cost + right.Cost;
                                    if (cost < bestSplit.Cost)
                                    {
                                        bestSplit.FeatureIndex = i;
                                        bestSplit.SplitIndex = j + 1;
                                        bestSplit.SplitValue = (observations.GetItemAt(index, i) + observations.GetItemAt(nextIndex, i)) * .5;
                                        bestSplit.LeftError = left.Cost;
                                        bestSplit.RightError = right.Cost;
                                        bestSplit.Cost = cost;
                                        bestSplit.CostImprovement = parentItem.Values.Cost - cost;
                                        bestSplit.LeftConstant = left.BestConstant;
                                        bestSplit.RightConstant = right.BestConstant;

                                        bestLeft = left.Copy();
                                        bestRight = right.Copy();
                                    }
                                }
                            }
                        }
                    }
                }

                if (bestSplit.FeatureIndex != -1)
                {
                    var node = bestSplit.GetNode();
                    nodeIndex++;
                    nodes.Add(node);

                    SetParentLeafIndex(nodeIndex, parentItem);
                    isLeaf = isLeaf || (bestSplit.CostImprovement < m_minimumInformationGain);
                    
                    if (!isLeaf)
                    {
                        var leftInSample = new bool[parentInSample.Length];
                        var rightInSample = new bool[parentInSample.Length];
                        var featureIndices = orderedElements[bestSplit.FeatureIndex];

                        for (int i = 0; i < parentInSample.Length; i++)
                        {
                            if(i < bestSplit.SplitIndex)
                            {
                                leftInSample[featureIndices[i]] = parentInSample[featureIndices[i]];
                            }
                            else
                            {
                                rightInSample[featureIndices[i]] = parentInSample[featureIndices[i]];
                            }
                        }

                        var depth = parentItem.Depth + 1;

                        stack.Enqueue(new GBMTreeCreationItem
                        {
                            Values = bestLeft.Copy(NodePositionType.Left),
                            InSample = leftInSample,
                            Depth = depth,
                            Parent = node
                        });

                        stack.Enqueue(new GBMTreeCreationItem
                        {
                            Values = bestRight.Copy(NodePositionType.Right),
                            InSample = rightInSample,
                            Depth = depth,
                            Parent = node
                        });
                    }
                }
            }

            return new GBMTree(nodes);
        }

        int NextIndexInSample(double[] y, bool[] parentInSample, int[] orderedIndices, int currentIndex)
        {
            var nextIndex = orderedIndices[currentIndex + 1];
            for (int i = currentIndex + 1; i < y.Length; i++)
            {
                if (parentInSample[orderedIndices[i]])
                {
                    nextIndex = orderedIndices[i];
                    break;
                }
            }
            return nextIndex;
        }

        void SetParentLeafIndex(int nodeIndex, GBMTreeCreationItem parentItem)
        {
            if (parentItem.Values.Position == NodePositionType.Left)
            {
                parentItem.Parent.LeftIndex = nodeIndex;
            }
            else if (parentItem.Values.Position == NodePositionType.Right)
            {
                parentItem.Parent.RightIndex = nodeIndex;
            }
        }
    }










 
}
