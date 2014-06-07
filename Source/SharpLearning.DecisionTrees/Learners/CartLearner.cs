using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.LeafValueFactories;
using SharpLearning.DecisionTrees.Models;
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
        readonly ILeafValueFactory m_leafValueFactory;
        
        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;

        double[] m_workTargets = new double[0];
        double[] m_workFeature = new double[0];
        int[] m_workIndices = new int[0];
        List<int> m_featureCandidates = new List<int>();
        int[] m_bestSplitWorkIndices = new int[0];
        int m_maximumTreeDepth;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="entropyMetric">The entropy metric used to calculate the best splits</param>
        /// <param name="featureCandidateSelector">The feature candidate selector used to decide which feature indices the learner can choose from at each split</param>
        /// <param name="leafValueFactory">The type of leaf created when no more splits can be made</param>
        public CartLearner(int minimumSplitSize, int maximumTreeDepth, double minimumInformationGain, IEntropyMetric entropyMetric, //ISplitFinder splitFinder, 
                           IFeatureCandidateSelector featureCandidateSelector, ILeafValueFactory leafValueFactory)
        {
            if (entropyMetric == null) { throw new ArgumentNullException("entropyMetric"); }
            //if (splitFinder == null) { throw new ArgumentNullException("splitFinder"); }
            if (featureCandidateSelector == null) { throw new ArgumentNullException("featureCandidateSelector"); }
            if (leafValueFactory == null) { throw new ArgumentNullException("leafValueFactory");}

            m_maximumTreeDepth = maximumTreeDepth;
            //m_splitFinder = splitFinder;
            m_entropyMetric = entropyMetric;
            m_featureCandidateSelector = featureCandidateSelector;
            m_leafValueFactory = leafValueFactory;
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
            Array.Resize(ref m_workTargets, indices.Length);
            Array.Resize(ref m_workFeature, indices.Length);
            Array.Resize(ref m_workIndices, indices.Length);
            Array.Resize(ref m_bestSplitWorkIndices, indices.Length);

            var allInterval = Interval1D.Create(0, indices.Length);
            indices.CopyTo(allInterval, m_workIndices);

            m_workIndices.IndexedCopy(targets, allInterval, m_workTargets);
            var rootEntropy = m_entropyMetric.Entropy(m_workTargets, allInterval);

            m_featureCandidateSelector.Select(observations.GetNumberOfColumns(), m_featureCandidates);

            var stack = new Stack<DecisionNodeCreationItem>(m_maximumTreeDepth);
            stack.Push(new DecisionNodeCreationItem(null, NodeType.Root, allInterval, rootEntropy));

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
                var node = new ContinousBinaryDecisionNode();

                if (bestSplitIndex >= 0 && bestInformationGain > m_minimumInformationGain)
                {

                    node.Parent = parentNode;
                    node.FeatureIndex = bestFeatureSplit.Index;
                    node.Value = bestFeatureSplit.Value;

                    stack.Push(new DecisionNodeCreationItem(node, NodeType.Right, bestRightInterval, bestRightEntropy));
                    stack.Push(new DecisionNodeCreationItem(node, NodeType.Left, bestLeftInterval, bestLeftEntropy));
                }
                else
                {
                    m_bestSplitWorkIndices.IndexedCopy(targets, parentInterval, m_workTargets);

                    node.Parent = parentNode;
                    node.FeatureIndex = -1;
                    node.Value = m_leafValueFactory.Calculate(m_workTargets, parentInterval);
                }

                switch (parentItem.NodeType)
                {
                    case NodeType.Root:
                        break;
                    case NodeType.Left:
                        parentNode.Right = node;
                        break;
                    case NodeType.Right:
                        parentNode.Left = node;
                        break;
                    default:
                        break;
                }
            }

            return root;
        }
    }
}
