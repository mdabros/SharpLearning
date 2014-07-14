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
using SharpLearning.DecisionTrees.SplitSearchers;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Trains a CART (Classification and Regression Tree) Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public class CartLearner
    {
        readonly IEntropyMetric m_entropyMetric;
        readonly ISplitSearcher m_splitSearcher;
        readonly IFeatureCandidateSelector m_featureCandidateSelector;
        readonly ILeafFactory m_leafFactory;
        
        readonly double m_minimumInformationGain;
        readonly int m_featuresPrSplit;

        double[] m_workTargets = new double[0];
        double[] m_workFeature = new double[0];
        double[] m_workWeights = new double[0];
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
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="featuresPrSplit">The number of features to be selected between at each split</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="entropyMetric">The entropy metric used to calculate the best splits</param>
        /// <param name="splitSearcher">The type of searcher used for finding the best features splits when learning the tree</param>
        /// <param name="featureCandidateSelector">The feature candidate selector used to decide which feature indices the learner can choose from at each split</param>
        /// <param name="leafFactory">The type of leaf created when no more splits can be made</param>
        public CartLearner(int maximumTreeDepth, int featuresPrSplit, double minimumInformationGain, IEntropyMetric entropyMetric,
            ISplitSearcher splitSearcher, IFeatureCandidateSelector featureCandidateSelector, ILeafFactory leafFactory)
        {
            if (entropyMetric == null) { throw new ArgumentNullException("entropyMetric"); }
            if (splitSearcher == null) { throw new ArgumentException("splitSearcher"); }
            if (featureCandidateSelector == null) { throw new ArgumentNullException("featureCandidateSelector"); }
            if (leafFactory == null) { throw new ArgumentNullException("leafValueFactory");}
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (featuresPrSplit < 1) { throw new ArgumentException("features pr split must be at least 1"); }
            
            m_entropyMetric = entropyMetric;
            m_maximumTreeDepth = maximumTreeDepth;
            m_featuresPrSplit = featuresPrSplit;
            m_splitSearcher = splitSearcher;
            m_featureCandidateSelector = featureCandidateSelector;
            m_leafFactory = leafFactory;
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
            return Learn(observations, targets, new double[0]);
        }

        /// <summary>
        /// Learns a CART decision tree from the provided observations and targets.
        /// Weights can be provided in order to weight each sample individually
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Learn(F64Matrix observations, double[] targets, double[] weights)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices, weights);
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
            return Learn(observations, targets, indices, new double[0]);
        }

        /// <summary>
        /// Learns a CART decision tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times. Weights can be provided in order to weight each sample individually
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights">Provide weights inorder to weigh each sample separetely</param>
        /// <returns></returns>
        public IBinaryDecisionNode Learn(F64Matrix observations, double[] targets, int[] indices, double[] weights)
        {
            using (var pinnedFeatures = observations.GetPinnedPointer())
            {
                return Learn(pinnedFeatures.View(), targets, indices, weights);
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
            return Learn(observations, targets, indices, new double[0]);
        }

        /// <summary>
        /// Learns a CART decision tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times. Weights can be provided in order to weight each sample individually
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights">Provide weights inorder to weigh each sample separetely</param>
        /// <returns></returns>
        public IBinaryDecisionNode Learn(F64MatrixView observations, double[] targets, int[] indices, double[] weights)
        {
            Array.Clear(m_variableImportance, 0, m_variableImportance.Length);

            Array.Resize(ref m_workTargets, indices.Length);
            Array.Resize(ref m_workFeature, indices.Length);
            Array.Resize(ref m_workIndices, indices.Length);


            Array.Resize(ref m_bestSplitWorkIndices, indices.Length);
            Array.Resize(ref m_variableImportance, observations.GetNumberOfColumns());
            m_featureCandidates.Clear();

            var allInterval = Interval1D.Create(0, indices.Length);

            indices.CopyTo(allInterval, m_workIndices);
            m_workIndices.IndexedCopy(targets, allInterval, m_workTargets);

            var rootEntropy = 0.0;

            if(weights.Length != 0)
            {
                Array.Resize(ref m_workWeights, indices.Length);
                m_workIndices.IndexedCopy(weights, allInterval, m_workWeights);
                rootEntropy = m_entropyMetric.Entropy(m_workTargets, m_workWeights, allInterval);
            }
            else
            {
                rootEntropy = m_entropyMetric.Entropy(m_workTargets, allInterval);
            }

            var uniqueValues = targets.Distinct().ToArray();

            m_featureCandidateSelector.Select(m_featuresPrSplit, observations.GetNumberOfColumns(), m_featureCandidates);

            var stack = new Stack<DecisionNodeCreationItem>(m_maximumTreeDepth);
            stack.Push(new DecisionNodeCreationItem(null, NodePositionType.Root, allInterval, rootEntropy, 0));

            var first = true;
            IBinaryDecisionNode root = null;
            
            while (stack.Count > 0)
            {
                var bestSplitResult = FindSplitResult.Initial();
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

                    if(weights.Length != 0)
                    {
                        m_workIndices.IndexedCopy(weights, parentInterval, m_workWeights);
                    }

                    var splitResult = m_splitSearcher.FindBestSplit(bestSplitResult, featureIndex, m_workFeature, m_workTargets,
                        m_workWeights, m_entropyMetric, parentInterval, parentEntropy);

                    if (splitResult.NewBestSplit)
                    {
                        bestSplitResult = splitResult;
                        m_workIndices.CopyTo(parentInterval, m_bestSplitWorkIndices);
                    }
                }

                m_bestSplitWorkIndices.CopyTo(parentInterval, m_workIndices);

                var bestSplitIndex = bestSplitResult.BestSplitIndex;
                var bestInformationGain = bestSplitResult.BestInformationGain;
                var bestLeftEntropy = bestSplitResult.LeftIntervalEntropy.Entropy;
                var bestRightEntropy = bestSplitResult.RightIntervalEntropy.Entropy;
                var bestLeftInterval = bestSplitResult.LeftIntervalEntropy.Interval;
                var bestRightInterval = bestSplitResult.RightIntervalEntropy.Interval;
                var bestFeatureSplit = bestSplitResult.BestFeatureSplit;

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
