using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.ImpurityCalculators;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Trains a Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public class DecisionTreeLearner
    {
        readonly ISplitSearcher m_splitSearcher;
        readonly IImpurityCalculator m_impurityCalculator;
        
        readonly double m_minimumInformationGain;
        readonly int m_featuresPrSplit;

        double[] m_workTargets = new double[0];
        double[] m_workFeature = new double[0];
        double[] m_workWeights = new double[0];
        int[] m_workIndices = new int[0];

        int[] m_allFeatureIndices = new int[0];
        int[] m_featureCandidates = new int[0];

        int[] m_bestSplitWorkIndices = new int[0];
        int m_maximumTreeDepth;

        Random m_random;
        bool m_featuresCandidatesSet = false;

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
        /// <param name="seed">Seed for feature selection if number of features pr split is not equal 
        /// to the total amount of features in observations. The features will be selected at random for each split</param>
        /// <param name="splitSearcher">The type of searcher used for finding the best features splits when learning the tree</param>
        /// <param name="impurityCalculator">Impurity calculator used to decide which split is optimal</param>
        public DecisionTreeLearner(int maximumTreeDepth, int featuresPrSplit, double minimumInformationGain, int seed,
            ISplitSearcher splitSearcher, IImpurityCalculator impurityCalculator)
        {
            if (splitSearcher == null) { throw new ArgumentException("splitSearcher"); }
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (featuresPrSplit < 1) { throw new ArgumentException("features pr split must be at least 1"); }
            if (impurityCalculator == null) { throw new ArgumentException("impurityCalculator"); }

            m_maximumTreeDepth = maximumTreeDepth;
            m_featuresPrSplit = featuresPrSplit;
            m_splitSearcher = splitSearcher;
            m_impurityCalculator = impurityCalculator;
            m_minimumInformationGain = minimumInformationGain;

            m_random = new Random(seed);
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public IBinaryDecisionNode Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets, new double[0]);
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets.
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
        /// Learns a decision tree from the provided observations and targets but limited to the observation indices provided by indices.
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
        /// Learns a decision tree from the provided observations and targets but limited to the observation indices provided by indices.
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
        /// Learns a decision tree from the provided observations and targets but limited to the observation indices provided by indices.
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
        /// Learns a decision tree from the provided observations and targets but limited to the observation indices provided by indices.
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

            var numberOfFeatures = observations.GetNumberOfColumns();
            Array.Resize(ref m_bestSplitWorkIndices, indices.Length);
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
            
            var uniqueValues = targets.Distinct().ToArray();
            
            m_impurityCalculator.Init(uniqueValues, m_workTargets, m_workWeights, allInterval);
            var rootImpurity = m_impurityCalculator.NodeImpurity();

            var stack = new Stack<DecisionNodeCreationItem>(m_maximumTreeDepth);
            stack.Push(new DecisionNodeCreationItem(null, NodePositionType.Root, allInterval, rootImpurity, 0));

            var first = true;
            IBinaryDecisionNode root = null;

            while (stack.Count > 0)
            {
                var bestSplitResult = SplitResult.Initial();
                var bestFeatureIndex = -1;
                var parentItem = stack.Pop();

                var parentInterval = parentItem.Interval;
                var parentNodeDepth = parentItem.NodeDepth;
                var parentNode = parentItem.Parent;
                var parentNodePositionType = parentItem.NodeType;
                var parentImpurity = parentItem.Impurity;
                                                          
                if (first && parentItem.Parent != null)
                {
                    root = parentNode;
                    first = false;
                }

                var isLeaf = (parentNodeDepth >= m_maximumTreeDepth);

                if(!isLeaf)
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

                        m_impurityCalculator.UpdateInterval(parentInterval);

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
                
                if(isLeaf)
                {
                    m_bestSplitWorkIndices.IndexedCopy(targets, parentInterval, m_workTargets);

                    if (weights.Length != 0)
                    {
                        m_bestSplitWorkIndices.IndexedCopy(weights, parentInterval, m_workWeights);
                    }
                    
                    m_impurityCalculator.UpdateInterval(parentInterval);
                    var value = m_impurityCalculator.LeafValue(); 

                    var leaf = new LeafBinaryDecisionNode(m_impurityCalculator.LeafProbabilities())
                    {
                        Parent = parentNode,
                        FeatureIndex = -1,
                        Value = value
                    };

                    parentNode.AddChild(parentNodePositionType, leaf);
                }
                else
                {
                    m_variableImportance[bestFeatureIndex] += bestSplitResult.ImpurityImprovement * parentInterval.Length / allInterval.Length;

                    var split = new ContinousBinaryDecisionNode
                    {
                        Parent = parentNode,
                        FeatureIndex = bestFeatureIndex,
                        Value = bestSplitResult.Threshold
                    };

                    var nodeDepth = parentNodeDepth + 1;

                    stack.Push(new DecisionNodeCreationItem(split, NodePositionType.Right, Interval1D.Create(bestSplitResult.SplitIndex, parentInterval.ToExclusive),
                        bestSplitResult.ImpurityRight, nodeDepth));
                    stack.Push(new DecisionNodeCreationItem(split, NodePositionType.Left, Interval1D.Create(parentInterval.FromInclusive, bestSplitResult.SplitIndex), 
                        bestSplitResult.ImpurityLeft, nodeDepth));

                    parentNode.AddChild(parentNodePositionType, split);
                }
            }

            if(root == null) // No valid split return single leaf result
            {
                m_impurityCalculator.UpdateInterval(allInterval);

                root = new LeafBinaryDecisionNode(m_impurityCalculator.LeafProbabilities())
                {
                    Parent = null,
                    FeatureIndex = -1,
                    Value = m_impurityCalculator.LeafValue()
                };
            }

            return root;
        }

        void SetNextFeatures(int totalNumberOfFeature)
        {
            if(m_featuresPrSplit != totalNumberOfFeature)
            {
                m_allFeatureIndices.Shuffle(m_random);
                Array.Copy(m_allFeatureIndices, m_featureCandidates, 
                    m_featuresPrSplit);
            }
            else if(!m_featuresCandidatesSet)
            {
                Array.Copy(m_allFeatureIndices, m_featureCandidates, 
                    m_allFeatureIndices.Length);

                m_featuresCandidatesSet = true;
            }
        }
    }
}
