using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.GradientBoost.Loss;
using SharpLearning.Threading;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.GradientBoost.GBMDecisionTree
{
    public sealed class GBMDecisionTreeLearner
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
        readonly int m_numberOfThreads;
        WorkerRunner m_threadedWorker;
        readonly IGradientBoostLoss m_loss;

        /// <summary>
        /// Fites a regression decision tree using a set presorted indices for each feature.
        /// </summary>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="loss">loss function used</param>
        /// <param name="numberOfThreads">Number of threads to use for paralization</param>
        public GBMDecisionTreeLearner(int maximumTreeDepth, int minimumSplitSize, double minimumInformationGain, IGradientBoostLoss loss, int numberOfThreads)
        {
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (loss == null) { throw new ArgumentNullException("loss"); }
            if (numberOfThreads < 1) { throw new ArgumentException("Number of threads must be at least 1"); }

            m_maximumTreeDepth = maximumTreeDepth;
            m_minimumSplitSize = minimumSplitSize;
            m_minimumInformationGain = minimumInformationGain;
            m_numberOfThreads = numberOfThreads;
            m_loss = loss;
        }

        /// <summary>
        /// Fites a regression decision tree using a set presorted indices for each feature.
        /// </summary>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        public GBMDecisionTreeLearner(int maximumTreeDepth = 2000, int minimumSplitSize = 1, double minimumInformationGain = 1E-6)
            : this(maximumTreeDepth, minimumSplitSize, minimumInformationGain, new GradientBoostSquaredLoss(), Environment.ProcessorCount)
        {
        }

        /// <summary>
        /// Fites a regression decision tree using a set presorted indices for each feature.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets">the original targets</param>
        /// <param name="residuals">the residuals for each boosting iteration</param>
        /// <param name="orderedElements">jagged array of sorted indices corresponding to each features</param>
        /// <param name="inSample">bool array containing the samples to use</param>
        /// <param name="s">sum</param>
        /// <param name="s2">sum of squares</param>
        /// <returns></returns>
        public GBMTree Learn(F64Matrix observations, double[] targets, double[] residuals, double[] predictions,
            int[][] orderedElements, bool[] inSample)
        {
            var rootValues = m_loss.InitSplit(targets, residuals, inSample);
            var bestConstant = rootValues.BestConstant;

            if(m_loss.UpdateLeafValues())
            {
                bestConstant = m_loss.UpdatedLeafValue(bestConstant,
                                    targets, predictions, inSample);
            }

            var root = new GBMNode()
            {
                FeatureIndex = -1,
                SplitValue = -1,
                LeftError = rootValues.Cost,
                RightError = rootValues.Cost,
                LeftConstant = bestConstant,
                RightConstant = bestConstant,
                SampleCount = rootValues.Samples
            };
            
            var nodes = new List<GBMNode> { root };

            var queue = new Queue<GBMTreeCreationItem>(100);
            queue.Enqueue(new GBMTreeCreationItem { Values = rootValues, InSample = inSample, Depth = 1 });

            var featureCount = observations.GetNumberOfColumns();
            var nodeIndex = 0;

            var splitResults = new ConcurrentBag<GBMSplitResult>();
            while (queue.Count > 0)
            {
                EmpyTySplitResults(splitResults);

                var parentItem = queue.Dequeue();
                var parentInSample = parentItem.InSample;

                var isLeaf = (parentItem.Depth >= m_maximumTreeDepth);

                var initBestSplit = new GBMSplit
                {
                    Depth = parentItem.Depth,
                    FeatureIndex = -1,
                    SplitIndex = -1,
                    SplitValue = -1,
                    Cost = double.MaxValue,
                    LeftConstant = -1,
                    RightConstant = -1

                };



                if (m_numberOfThreads == 1)
                {
                    for (int i = 0; i < featureCount; i++)
                    {
                        FindBestSplit(observations, residuals, targets, predictions, orderedElements,
                            parentItem, parentInSample, i, splitResults);

                    }
                }
                else
                {
                    var workItems = new ConcurrentQueue<int>();
                    for (int i = 0; i < featureCount; i++)
                    {
                        workItems.Enqueue(i);
                    }

                    Action workAction = () => SplitWorker(observations, residuals, targets, predictions, orderedElements, parentItem,
                            parentInSample, workItems, splitResults);

                    var workers = new List<Action>();
                    for (int i = 0; i < m_numberOfThreads; i++)
                    {
                        workers.Add(workAction);
                    }

                    m_threadedWorker = new WorkerRunner(workers);
                    m_threadedWorker.Run();
                }

                var bestSplitResult = new GBMSplitResult { BestSplit = initBestSplit, Left = GBMSplitInfo.NewEmpty(), Right = GBMSplitInfo.NewEmpty() };
                if (splitResults.Count != 0)
                {
                    // alternative to for finding bestsplit. gives slightly different results. probably due to order.
                    //GBMSplitResult result;
                    //while (splitResults.TryTake(out result))
                    //{
                    //    if (result.BestSplit.Cost < bestSplitResult.BestSplit.Cost)
                    //    {
                    //        bestSplitResult = result;
                    //    }
                    //}
                    bestSplitResult = splitResults.OrderBy(r => r.BestSplit.Cost).First();

                    var node = bestSplitResult.BestSplit.GetNode();
                    nodeIndex++;
                    nodes.Add(node);

                    SetParentLeafIndex(nodeIndex, parentItem);
                    isLeaf = isLeaf || (bestSplitResult.BestSplit.CostImprovement < m_minimumInformationGain);

                    if (!isLeaf)
                    {
                        var leftInSample = new bool[parentInSample.Length];
                        var rightInSample = new bool[parentInSample.Length];
                        var featureIndices = orderedElements[bestSplitResult.BestSplit.FeatureIndex];

                        for (int i = 0; i < parentInSample.Length; i++)
                        {
                            if (i < bestSplitResult.BestSplit.SplitIndex)
                            {
                                leftInSample[featureIndices[i]] = parentInSample[featureIndices[i]];
                            }
                            else
                            {
                                rightInSample[featureIndices[i]] = parentInSample[featureIndices[i]];
                            }
                        }

                        if (m_loss.UpdateLeafValues())
                        {
                            node.LeftConstant = m_loss.UpdatedLeafValue(node.LeftConstant,
                                targets, predictions, leftInSample);

                            node.RightConstant = m_loss.UpdatedLeafValue(node.RightConstant,
                                targets, predictions, rightInSample);
                        }

                        var depth = parentItem.Depth + 1;

                        queue.Enqueue(new GBMTreeCreationItem
                        {
                            Values = bestSplitResult.Left.Copy(NodePositionType.Left),
                            InSample = leftInSample,
                            Depth = depth,
                            Parent = node
                        });

                        queue.Enqueue(new GBMTreeCreationItem
                        {
                            Values = bestSplitResult.Right.Copy(NodePositionType.Right),
                            InSample = rightInSample,
                            Depth = depth,
                            Parent = node
                        });
                    }
                    else
                    {
                        if (m_loss.UpdateLeafValues())
                        {
                            var leftInSample = new bool[parentInSample.Length];
                            var rightInSample = new bool[parentInSample.Length];
                            var featureIndices = orderedElements[bestSplitResult.BestSplit.FeatureIndex];


                            for (int i = 0; i < parentInSample.Length; i++)
                            {
                                if (i < bestSplitResult.BestSplit.SplitIndex)
                                {
                                    leftInSample[featureIndices[i]] = parentInSample[featureIndices[i]];
                                }
                                else
                                {
                                    rightInSample[featureIndices[i]] = parentInSample[featureIndices[i]];
                                }
                            }

                            node.LeftConstant = m_loss.UpdatedLeafValue(node.LeftConstant,
                                targets, predictions, leftInSample);

                            node.RightConstant = m_loss.UpdatedLeafValue(node.RightConstant,
                                targets, predictions, rightInSample);
                        }
                    }
                }
            }

            return new GBMTree(nodes);
        }

        private static void EmpyTySplitResults(ConcurrentBag<GBMSplitResult> splitResults)
        {
            GBMSplitResult result;
            while (splitResults.TryTake(out result)) ;
        }

        void SplitWorker(F64Matrix observations, double[] residuals, double[] targets, double[] predictions, int[][] orderedElements, 
            GBMTreeCreationItem parentItem, bool[] parentInSample, ConcurrentQueue<int> featureIndices, ConcurrentBag<GBMSplitResult> results)
        {
            int featureIndex = -1;
            while (featureIndices.TryDequeue(out featureIndex))
            {
                FindBestSplit(observations, residuals, targets, predictions, orderedElements, 
                    parentItem, parentInSample, featureIndex, results);
            }
        }

        void FindBestSplit(F64Matrix observations, double[] residuals, double[] targets, double[] predictions, int[][] orderedElements, 
            GBMTreeCreationItem parentItem, bool[] parentInSample, int featureIndex, ConcurrentBag<GBMSplitResult> results)
        {
            var bestSplit = new GBMSplit
            {
                Depth = parentItem.Depth,
                FeatureIndex = -1,
                SplitIndex = -1,
                SplitValue = -1,
                Cost = double.MaxValue,
                LeftConstant = -1,
                RightConstant = -1,
                SampleCount = parentItem.Values.Samples
            };

            var bestLeft = GBMSplitInfo.NewEmpty();
            var bestRight = GBMSplitInfo.NewEmpty();

            var left = GBMSplitInfo.NewEmpty();
            var right = parentItem.Values.Copy(NodePositionType.Right);

            var orderedIndices = orderedElements[featureIndex];
            var j = NextAllowedIndex(0, orderedIndices, parentInSample);
            var currentIndex = orderedIndices[j];
      
            m_loss.UpdateSplitConstants(ref left, ref right, targets[currentIndex], residuals[currentIndex]);

            var previousValue = observations.GetItemAt(currentIndex, featureIndex);
                      
            while(right.Samples > 0)
            {
                j = NextAllowedIndex(j + 1, orderedIndices, parentInSample);
                currentIndex = orderedIndices[j];
                var currentValue = observations.GetItemAt(currentIndex, featureIndex);

                if (Math.Min(left.Samples, right.Samples) >= m_minimumSplitSize)
                {
                    if (previousValue != currentValue)
                    {
                        var cost = left.Cost + right.Cost;
                        if (cost < bestSplit.Cost)
                        {
                            bestSplit.FeatureIndex = featureIndex;
                            bestSplit.SplitIndex = j;
                            bestSplit.SplitValue = (previousValue + currentValue) * .5;
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

                m_loss.UpdateSplitConstants(ref left, ref right, targets[currentIndex], residuals[currentIndex]);
                previousValue = currentValue;
            }

            if(bestSplit.FeatureIndex != -1)
            {
                results.Add(new GBMSplitResult { BestSplit = bestSplit, Left = bestLeft, Right = bestRight });
            }
        }

        int NextAllowedIndex(int start, int[] orderedIndexes, bool[] inSample)
        {

            for (int i = start; i < orderedIndexes.Length; i++)
            {
                if (inSample[orderedIndexes[i]])
                {
                    return i;
                }
            }
            return (orderedIndexes.Length + 1);
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
