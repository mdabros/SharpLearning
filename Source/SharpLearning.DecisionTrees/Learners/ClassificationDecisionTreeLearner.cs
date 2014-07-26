using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Trains a Classification Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public sealed class ClassificationDecisionTreeLearner : DecisionTreeLearner
    {
        /// <summary>
        /// Trains a Classification Decision tree
        /// http://en.wikipedia.org/wiki/Decision_tree_learning
        /// </summary>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="featuresPrSplit">The number of features to be selected between at each split</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="seed">Seed for feature selection if number of features pr split is not equal 
        /// to the total amount of features in observations. The features will be selected at random for each split</param>
        public ClassificationDecisionTreeLearner(int maximumTreeDepth=2000, int minimumSplitSize=1, int featuresPrSplit=0, double minimumInformationGain=0.000001, int seed=42)
            : base(maximumTreeDepth, featuresPrSplit, minimumInformationGain, seed, new OnlyUniqueThresholdsSplitSearcher(minimumSplitSize),
                   new GiniClasificationImpurityCalculator())
        {
        }
                
        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public new ClassificationDecisionTreeModel Learn(F64Matrix observations, double[] targets)
        {
            return new ClassificationDecisionTreeModel(base.Learn(observations, targets), m_variableImportance);
        }

        /// <summary>
        /// Learns a classification tree from the provided observations and targets. 
        /// Weights can be provided in order to weight each sample individually
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public new ClassificationDecisionTreeModel Learn(F64Matrix observations, double[] targets, double[] weights)
        {
            return new ClassificationDecisionTreeModel(base.Learn(observations, targets, weights), m_variableImportance);
        }

        /// <summary>
        /// Learns a classification tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new ClassificationDecisionTreeModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return new ClassificationDecisionTreeModel(base.Learn(observations, targets, indices), m_variableImportance);
        }

        /// <summary>
        /// Learns a classification tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times. Weights can be provided in order to weight each sample individually.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new ClassificationDecisionTreeModel Learn(F64Matrix observations, double[] targets, int[] indices, double[] weights)
        {
            return new ClassificationDecisionTreeModel(base.Learn(observations, targets, indices, weights), m_variableImportance);
        }

        /// <summary>
        /// Learns a classification tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new ClassificationDecisionTreeModel Learn(F64MatrixView observations, double[] targets, int[] indices)
        {
            return new ClassificationDecisionTreeModel(base.Learn(observations, targets, indices), m_variableImportance);
        }

        /// <summary>
        /// Learns a classification tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new ClassificationDecisionTreeModel Learn(F64MatrixView observations, double[] targets, int[] indices, double[] weights)
        {
            return new ClassificationDecisionTreeModel(base.Learn(observations, targets, indices, weights), m_variableImportance);
        }

    }
}
