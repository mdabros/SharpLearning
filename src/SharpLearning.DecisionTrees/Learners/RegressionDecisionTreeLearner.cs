using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.TreeBuilders;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Trains a Regression Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public sealed class RegressionDecisionTreeLearner : DecisionTreeLearner,
        IIndexedLearner<double>, ILearner<double>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="featuresPrSplit">The number of features to be selected between at each split</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="seed">Seed for feature selection if number of features pr split is not equal 
        /// to the total amount of features in observations. The features will be selected at random for each split</param>
        public RegressionDecisionTreeLearner(int maximumTreeDepth=2000, 
            int minimumSplitSize=1, 
            int featuresPrSplit=0, 
            double minimumInformationGain=0.000001, 
            int seed=42)
            : base(new DepthFirstTreeBuilder(maximumTreeDepth, featuresPrSplit, minimumInformationGain, seed, 
                new OnlyUniqueThresholdsSplitSearcher(minimumSplitSize), 
                new RegressionImpurityCalculator()))
        {
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64Matrix observations, double[] targets)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets));
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64Matrix observations, double[] targets, 
            double[] weights)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, weights));
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64Matrix observations, double[] targets, 
            int[] indices)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, indices));
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64Matrix observations, double[] targets, 
            int[] indices, double[] weights)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, indices, weights));
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64MatrixView observations, double[] targets, 
            int[] indices)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, indices));
        }

        /// <summary>
        /// Learns a regression tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public new RegressionDecisionTreeModel Learn(F64MatrixView observations, double[] targets, 
            int[] indices, double[] weights)
        {
            return new RegressionDecisionTreeModel(base.Learn(observations, targets, indices, weights));
        }

        /// <summary>
        /// Private explicit interface implementation for indexed learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<double> IIndexedLearner<double>.Learn(
            F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

        /// <summary>
        /// Private explicit interface implementation for learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(
            F64Matrix observations, double[] targets) => Learn(observations, targets);
    }
}
