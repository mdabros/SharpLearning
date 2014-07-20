using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.FeatureCandidateSelectors;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Trains a Regression Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public sealed class RegressionDecisionTreeLearner : DecisionTreeLearner
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="featuresPrSplit">The number of features to be selected between at each split</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        public RegressionDecisionTreeLearner(int minimumSplitSize, int maximumTreeDepth, int featuresPrSplit, double minimumInformationGain)
            : base(maximumTreeDepth, featuresPrSplit, minimumInformationGain, new LinearSplitSearcher(minimumSplitSize), 
                new RegressionImpurityCalculator(), new AllFeatureCandidateSelector())
        {
        }

        public new RegressionDecisionTreeModel Learn(F64Matrix observations, double[] targets)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets), m_variableImportance);
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64Matrix observations, double[] targets, double[] weights)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, weights), m_variableImportance);
        }

        public new RegressionDecisionTreeModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, indices), m_variableImportance);
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64Matrix observations, double[] targets, int[] indices, double[] weights)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, indices, weights), m_variableImportance);
        }

        public new RegressionDecisionTreeModel Learn(F64MatrixView observations, double[] targets, int[] indices)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, indices), m_variableImportance);
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64MatrixView observations, double[] targets, int[] indices, double[] weights)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, indices, weights), m_variableImportance);
        }
    }
}
