using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.DecisionTrees.SplitSearchers;

namespace SharpLearning.DecisionTrees.TreeBuilders
{
    /// <summary>
    /// Builds a decision tree in a best first manner. 
    /// This method enables maximum leaf nodes to be set. 
    /// </summary>
    public sealed class BestFirstTreeBuilder : ITreeBuilder
    {
        readonly ISplitSearcher m_splitSearcher;
        readonly IImpurityCalculator m_impurityCalculator;

        readonly double m_minimumInformationGain;
        readonly int m_maximumTreeDepth;
        readonly int m_maximumLeafCount;
        readonly Random m_random;

        int m_featuresPrSplit;

        double[] m_workTargets = new double[0];
        double[] m_workFeature = new double[0];
        double[] m_workWeights = new double[0];
        int[] m_workIndices = new int[0];

        int[] m_allFeatureIndices = new int[0];
        int[] m_featureCandidates = new int[0];

        int[] m_bestSplitWorkIndices = new int[0];
        bool m_featuresCandidatesSet = false;

        // Variable importances are based on the work each variable does (information gain).
        // the scores at each split is scaled by the amount of data the node splits
        // if a node splits on 30% of the total data it will add
        // informationGain * 0.3 to its importance score.
        // Based on this explanation:
        // http://www.salford-systems.com/videos/tutorials/how-to/variable-importance-in-cart
        double[] m_variableImportance = new double[0];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="maximumLeafCount">The maximal allowed leaf nodes in the tree</param>
        /// <param name="featuresPrSplit">The number of features to be selected between at each split. 
        /// 0 means use all available features</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="seed">Seed for feature selection if number of features pr split is not equal 
        /// to the total amount of features in observations. The features will be selected at random for each split</param>
        /// <param name="splitSearcher">The type of searcher used for finding the best features splits when learning the tree</param>
        /// <param name="impurityCalculator">Impurity calculator used to decide which split is optimal</param>
        public BestFirstTreeBuilder(int maximumTreeDepth, 
            int maximumLeafCount, 
            int featuresPrSplit, 
            double minimumInformationGain, 
            int seed,
            ISplitSearcher splitSearcher, 
            IImpurityCalculator impurityCalculator)
        {
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (maximumLeafCount <= 1) { throw new ArgumentException("maximum leaf count must be larger than 1"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (featuresPrSplit < 0) { throw new ArgumentException("features pr split must be at least 0"); }
            m_splitSearcher = splitSearcher ?? throw new ArgumentException(nameof(splitSearcher));
            m_impurityCalculator = impurityCalculator ?? throw new ArgumentException(nameof(impurityCalculator));

            m_maximumTreeDepth = maximumTreeDepth;
            m_maximumLeafCount = maximumLeafCount;
            m_featuresPrSplit = featuresPrSplit;
            m_minimumInformationGain = minimumInformationGain;

            m_random = new Random(seed);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public BinaryTree Build(F64MatrixView observations, double[] targets, int[] indices, double[] weights)
        {
            Array.Clear(m_variableImportance, 0, m_variableImportance.Length);

            Array.Resize(ref m_workTargets, indices.Length);
            Array.Resize(ref m_workFeature, indices.Length);
            Array.Resize(ref m_workIndices, indices.Length);

            var numberOfFeatures = observations.ColumnCount;

            if (m_featuresPrSplit == 0)
            {
                m_featuresPrSplit = numberOfFeatures;
            }

            Array.Resize(ref m_bestSplitWorkIndices, indices.Length);
            m_bestSplitWorkIndices.Clear();
            Array.Resize(ref m_variableImportance, numberOfFeatures);
            Array.Resize(ref m_allFeatureIndices, numberOfFeatures);
            Array.Resize(ref m_featureCandidates, m_featuresPrSplit);

            m_featuresCandidatesSet = false;

            for (int i = 0; i < m_allFeatureIndices.Length; i++)
            {
                m_allFeatureIndices[i] = i;
            }

            var allInterval = Interval1D.Create(0, indices.Length);

            indices.CopyTo(allInterval, m_workIndices);
            m_workIndices.IndexedCopy(targets, allInterval, m_workTargets);

            if (weights.Length != 0)
            {
                Array.Resize(ref m_workWeights, indices.Length);
                m_workIndices.IndexedCopy(weights, allInterval, m_workWeights);
            }

            var targetNames = targets.Distinct().ToArray();

            m_impurityCalculator.Init(targetNames, m_workTargets, m_workWeights, allInterval);
            var rootImpurity = m_impurityCalculator.NodeImpurity();

            var nodes = new List<Node>();
            var probabilities = new List<double[]>();

            var queue = new Queue<DecisionNodeCreationItem>(100);
            queue.Enqueue(new DecisionNodeCreationItem(0, NodePositionType.Root, allInterval, rootImpurity, 0));

            var first = true;
            var currentNodeIndex = 0;
            var currentLeafProbabilityIndex = 0;

            var maximumSplits = m_maximumLeafCount - 1;

            while (queue.Count > 0)
            {
                var bestSplitResult = SplitResult.Initial();
                var bestFeatureIndex = -1;
                var parentItem = queue.Dequeue();

                var parentInterval = parentItem.Interval;
                var parentNodeDepth = parentItem.NodeDepth;
                Node parentNode = Node.Default();

                if (nodes.Count != 0)
                {
                    parentNode = nodes[parentItem.ParentIndex];
                }

                var parentNodePositionType = parentItem.NodeType;
                var parentImpurity = parentItem.Impurity;

                if (first && parentNode.FeatureIndex != -1)
                {
                    nodes[0] = new Node(parentNode.FeatureIndex,
                        parentNode.Value, -1, -1, parentNode.NodeIndex, parentNode.LeafProbabilityIndex);

                    first = false;
                }

                var isLeaf = (parentNodeDepth >= m_maximumTreeDepth || maximumSplits <= 0);

                if (!isLeaf)
                {
                    SetNextFeatures(numberOfFeatures);

                    foreach (var featureIndex in m_featureCandidates)
                    {
                        m_workIndices.IndexedCopy(observations.ColumnView(featureIndex), parentInterval, m_workFeature);
                        m_workFeature.SortWith(parentInterval, m_workIndices);
                        m_workIndices.IndexedCopy(targets, parentInterval, m_workTargets);

                        if (weights.Length != 0)
                        {
                            m_workIndices.IndexedCopy(weights, parentInterval, m_workWeights);
                        }

                        var splitResult = m_splitSearcher.FindBestSplit(m_impurityCalculator, m_workFeature,
                            m_workTargets, parentInterval, parentImpurity);

                        if (splitResult.ImpurityImprovement > bestSplitResult.ImpurityImprovement)
                        {
                            bestSplitResult = splitResult;
                            m_workIndices.CopyTo(parentInterval, m_bestSplitWorkIndices);
                            bestFeatureIndex = featureIndex;
                        }
                    }

                    isLeaf = isLeaf || (bestSplitResult.SplitIndex < 0);
                    isLeaf = isLeaf || (bestSplitResult.ImpurityImprovement < m_minimumInformationGain);

                    m_bestSplitWorkIndices.CopyTo(parentInterval, m_workIndices);
                }

                if (isLeaf)
                {
                    m_bestSplitWorkIndices.IndexedCopy(targets, parentInterval, m_workTargets);

                    if (weights.Length != 0)
                    {
                        m_bestSplitWorkIndices.IndexedCopy(weights, parentInterval, m_workWeights);
                    }

                    m_impurityCalculator.UpdateInterval(parentInterval);
                    var value = m_impurityCalculator.LeafValue();

                    var leaf = new Node(-1, value, -1, -1,
                        currentNodeIndex++, currentLeafProbabilityIndex++);

                    probabilities.Add(m_impurityCalculator.LeafProbabilities());

                    nodes.Add(leaf);
                    nodes.UpdateParent(parentNode, leaf, parentNodePositionType);
                }
                else
                {
                    maximumSplits--;
                    m_variableImportance[bestFeatureIndex] += bestSplitResult.ImpurityImprovement * parentInterval.Length / allInterval.Length;

                    var split = new Node(bestFeatureIndex, bestSplitResult.Threshold, -1, -1,
                        currentNodeIndex++, -1);

                    nodes.Add(split);
                    nodes.UpdateParent(parentNode, split, parentNodePositionType);

                    var nodeDepth = parentNodeDepth + 1;

                    queue.Enqueue(new DecisionNodeCreationItem(split.NodeIndex, NodePositionType.Right,
                        Interval1D.Create(bestSplitResult.SplitIndex, parentInterval.ToExclusive),
                        bestSplitResult.ImpurityRight, nodeDepth));

                    queue.Enqueue(new DecisionNodeCreationItem(split.NodeIndex, NodePositionType.Left,
                        Interval1D.Create(parentInterval.FromInclusive, bestSplitResult.SplitIndex),
                        bestSplitResult.ImpurityLeft, nodeDepth));
                }
            }

            if (first) // No valid split return single leaf result
            {
                m_impurityCalculator.UpdateInterval(allInterval);

                var leaf = new Node(-1, m_impurityCalculator.LeafValue(), -1, -1,
                    currentNodeIndex++, currentLeafProbabilityIndex++);

                probabilities.Add(m_impurityCalculator.LeafProbabilities());

                nodes.Clear();
                nodes.Add(leaf);
            }

            return new BinaryTree(nodes, probabilities, targetNames, 
                m_variableImportance.ToArray());
        }

        void SetNextFeatures(int totalNumberOfFeature)
        {
            if (m_featuresPrSplit != totalNumberOfFeature)
            {
                m_allFeatureIndices.Shuffle(m_random);
                Array.Copy(m_allFeatureIndices, m_featureCandidates,
                    m_featuresPrSplit);
            }
            else if (!m_featuresCandidatesSet)
            {
                Array.Copy(m_allFeatureIndices, m_featureCandidates,
                    m_allFeatureIndices.Length);

                m_featuresCandidatesSet = true;
            }
        }
    }
}
