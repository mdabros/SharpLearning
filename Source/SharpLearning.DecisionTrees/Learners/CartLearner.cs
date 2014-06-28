using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.LeafFactories;
using SharpLearning.DecisionTrees.FeatureCandidateSelectors;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Metrics.Entropy;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Trains a CART (Classification and Regression Tree) Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public class CartLearner
    {
        readonly IEntropyMetric m_entropyMetric;
        //readonly ISplitFinder m_splitFinder;
        readonly IFeatureCandidateSelector m_featureCandidateSelector;
        readonly ILeafFactory m_leafFactory;
        
        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;
        readonly int m_featuresPrSplit;

        double[] m_workTargets = new double[0];
        double[] m_workFeature = new double[0];
        int[] m_workIndices = new int[0];
        List<int> m_featureCandidates = new List<int>();
        int[] m_bestSplitWorkIndices = new int[0];
        int m_maximumTreeDepth;

        // Variable importances are based on the work each variable does (information gain).
        // the scores at each split is scaled by the amount of data the node splits
        // if a node splits on 30% of the total data it will add
        // informationGain * 0.3 to its importance score.
        // Based on this explanation:
        // http://www.salford-systems.com/videos/tutorials/how-to/variable-importance-in-cart
        public double[] m_variableImportance = new double[0];

        /// <summary>
        /// 
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="featuresPrSplit">The number of features to be selected between at each split</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="entropyMetric">The entropy metric used to calculate the best splits</param>
        /// <param name="featureCandidateSelector">The feature candidate selector used to decide which feature indices the learner can choose from at each split</param>
        /// <param name="leafFactory">The type of leaf created when no more splits can be made</param>
        public CartLearner(int minimumSplitSize, int maximumTreeDepth, int featuresPrSplit, double minimumInformationGain, IEntropyMetric entropyMetric, //ISplitFinder splitFinder, 
                           IFeatureCandidateSelector featureCandidateSelector, ILeafFactory leafFactory)
        {
            if (entropyMetric == null) { throw new ArgumentNullException("entropyMetric"); }
            //if (splitFinder == null) { throw new ArgumentNullException("splitFinder"); }
            if (featureCandidateSelector == null) { throw new ArgumentNullException("featureCandidateSelector"); }
            if (leafFactory == null) { throw new ArgumentNullException("leafValueFactory");}
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (featuresPrSplit < 1) { throw new ArgumentException("features pr split must be at least 1"); }

            m_maximumTreeDepth = maximumTreeDepth;
            m_featuresPrSplit = featuresPrSplit;
            //m_splitFinder = splitFinder;
            m_entropyMetric = entropyMetric;
            m_featureCandidateSelector = featureCandidateSelector;
            m_leafFactory = leafFactory;
            m_minimumSplitSize = minimumSplitSize;
            m_minimumInformationGain = minimumInformationGain;
        }

        /// <summary>
        /// Learns a CART decision tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a CART decision tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            using (var pinnedFeatures = observations.GetPinnedPointer())
            {
                return Learn(pinnedFeatures.View(), targets, indices);
            }
        }

        /// <summary>
        /// Learns a CART decision tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Learn(F64MatrixView observations, double[] targets, int[] indices)
        {
            Array.Clear(m_variableImportance, 0, m_variableImportance.Length);

            Array.Resize(ref m_workTargets, indices.Length);
            Array.Resize(ref m_workFeature, indices.Length);
            Array.Resize(ref m_workIndices, indices.Length);
            Array.Resize(ref m_bestSplitWorkIndices, indices.Length);
            Array.Resize(ref m_variableImportance, observations.GetNumberOfColumns());
            m_featureCandidates.Clear();

            var uniqueValues = targets.Distinct().ToArray();

            var allInterval = Interval1D.Create(0, indices.Length);
            indices.CopyTo(allInterval, m_workIndices);

            m_workIndices.IndexedCopy(targets, allInterval, m_workTargets);
            var rootEntropy = m_entropyMetric.Entropy(m_workTargets, allInterval);

            m_featureCandidateSelector.Select(m_featuresPrSplit, observations.GetNumberOfColumns(), m_featureCandidates);

            var stack = new Stack<DecisionNodeCreationItem>(m_maximumTreeDepth);
            stack.Push(new DecisionNodeCreationItem(null, NodePositionType.Root, allInterval, rootEntropy, 0));

            var first = true;
            IBinaryDecisionNode root = null;


            while (stack.Count > 0)
            {
                int bestSplitIndex = -1;
                double bestInformationGain = 0;
                double bestLeftEntropy = 0.0;
                double bestRightEntropy = 0.0;
                Interval1D bestLeftInterval = Interval1D.Create(0, 1);
                Interval1D bestRightInterval = Interval1D.Create(0, 1);

                var bestFeatureSplit = new FeatureSplit();
                
                var parentItem = stack.Pop();

                var parentInterval = parentItem.Interval;
                var parentNode = parentItem.Parent;
                var parentEntropy = parentItem.Entropy;
                var parentNodeDepth = parentItem.NodeDepth;
                var parentNodePositionType = parentItem.NodeType;

                if (first && parentItem.Parent != null)
                {
                    root = parentNode;
                    first = false;
                }

                foreach (var featureIndex in m_featureCandidates)
                {
                    m_workIndices.IndexedCopy(observations.ColumnView(featureIndex), parentInterval, m_workFeature);
                    m_workFeature.SortWith(parentInterval, m_workIndices);
                    m_workIndices.IndexedCopy(targets, parentInterval, m_workTargets);

                    bool newBestSplit = false;

                    int prevSplit = parentInterval.FromInclusive;
                    var prevValue = m_workFeature[prevSplit];
                    var prevTarget = m_workTargets[prevSplit];

                    for (int j = prevSplit + 1; j < parentInterval.ToExclusive; j++)
                    {
                        var currentValue = m_workFeature[j];
                        var currentTarget = m_workTargets[j];
                        if (prevValue != currentValue && prevTarget != currentTarget)
                        {
                            var currentSplit = j;
                            var leftSize = currentSplit - parentInterval.FromInclusive;
                            var rightSize = parentInterval.ToExclusive - currentSplit;

                            if (Math.Min(leftSize, rightSize) >= m_minimumSplitSize)
                            {
                                var leftInterval = Interval1D.Create(parentInterval.FromInclusive, currentSplit);
                                var rightInterval = Interval1D.Create(currentSplit, parentInterval.ToExclusive);

                                var leftEntropy = m_entropyMetric.Entropy(m_workTargets, leftInterval);
                                var rightEntropy = m_entropyMetric.Entropy(m_workTargets, rightInterval);
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

                    if (newBestSplit)
                    {
                        m_workIndices.CopyTo(parentInterval, m_bestSplitWorkIndices);
                    }
                }

                m_bestSplitWorkIndices.CopyTo(parentInterval, m_workIndices);

                if (bestSplitIndex >= 0 && bestInformationGain > m_minimumInformationGain && m_maximumTreeDepth > parentNodeDepth)
                {
                    m_variableImportance[bestFeatureSplit.Index] += bestInformationGain * parentInterval.Length / allInterval.Length;

                    var split = new ContinousBinaryDecisionNode
                    {
                        Parent = parentNode,
                        FeatureIndex = bestFeatureSplit.Index,
                        Value = bestFeatureSplit.Value
                    };

                    var nodeDepth = parentNodeDepth + 1;

                    stack.Push(new DecisionNodeCreationItem(split, NodePositionType.Right, bestRightInterval, bestRightEntropy, nodeDepth));
                    stack.Push(new DecisionNodeCreationItem(split, NodePositionType.Left, bestLeftInterval, bestLeftEntropy, nodeDepth));

                    parentNode.AddChild(parentNodePositionType, split);
                }
                else
                {
                    m_bestSplitWorkIndices.IndexedCopy(targets, parentInterval, m_workTargets);
                    
                    var leaf = m_leafFactory.Create(parentNode, m_workTargets, uniqueValues, parentInterval);

                    parentNode.AddChild(parentNodePositionType, leaf);
                }
            }

            if(root == null) // No valid split return single leaf result
            {
                root = m_leafFactory.Create(null, targets, uniqueValues);
            }

            return root;
        }
    }
}
