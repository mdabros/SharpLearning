using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.LeafFactories;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Metrics.Entropy;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Trains a CART Classification Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public sealed class ClassificationCartLearner : CartLearner
    {
        /// <summary>
        /// Trains a CART Classification Decision tree
        /// http://en.wikipedia.org/wiki/Decision_tree_learning
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="maxTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        public ClassificationCartLearner(int minimumSplitSize, int maximumTreeDepth, double minimumInformationGain)
            : base(minimumSplitSize, maximumTreeDepth, minimumInformationGain, new GiniImpurityMetric(), //new LinearSplitFinder(),
                   new AllFeatureCandidateSelector(), new ClassificationLeafFactory())
        {
        }

        /// <summary>
        /// Learns a CART classification tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public new ClassificationCartModel Learn(F64Matrix observations, double[] targets)
        {
            return new ClassificationCartModel(base.Learn(observations, targets), m_variableImportance);
        }

        /// <summary>
        /// Learns a CART classification tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new ClassificationCartModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return new ClassificationCartModel(base.Learn(observations, targets, indices), m_variableImportance);
        }

        /// <summary>
        /// Learns a CART classification tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public new ClassificationCartModel Learn(F64MatrixView observations, double[] targets, int[] indices)
        {
            return new ClassificationCartModel(base.Learn(observations, targets, indices), m_variableImportance);
        }
    }
}
