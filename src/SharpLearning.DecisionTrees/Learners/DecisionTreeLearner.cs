using System;
using System.Linq;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.DecisionTrees.TreeBuilders;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Learns a Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public unsafe class DecisionTreeLearner
    {
        readonly ITreeBuilder m_treeBuilder;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="treeBuilder"></param>
        public DecisionTreeLearner(ITreeBuilder treeBuilder)
        {
            m_treeBuilder = treeBuilder ?? throw new ArgumentNullException(nameof(treeBuilder));
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public BinaryTree Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets, new double[0]);
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets.
        /// Weights can be provided in order to weight each sample individually
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public BinaryTree Learn(F64Matrix observations, double[] targets, double[] weights)
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
        public BinaryTree Learn(F64Matrix observations, double[] targets, int[] indices)
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
        /// <param name="weights">Provide weights in order to weigh each sample separately</param>
        /// <returns></returns>
        public BinaryTree Learn(F64Matrix observations, double[] targets, int[] indices, double[] weights)
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
        public BinaryTree Learn(F64MatrixView observations, double[] targets, int[] indices)
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
        /// <param name="weights">Provide weights in order to weigh each sample separately</param>
        /// <returns></returns>
        public BinaryTree Learn(F64MatrixView observations, double[] targets, int[] indices, double[] weights)
        {
            Checks.VerifyObservationsAndTargets(observations, targets);
            Checks.VerifyIndices(indices, observations, targets);

            // Verify weights dimensions. Currently sample weights is supported by DecisionTreeLearner.
            // Hence, the check is not added to the general checks.
            if (weights.Length != 0)
            {
                if (weights.Length != targets.Length || weights.Length != observations.RowCount)
                {
                    throw new ArgumentException($"Weights length differ from observation row count and target length. " +
                        $"Weights: {weights.Length}, observation: {observations.RowCount}, targets: {targets.Length}");
                }
            }
            return m_treeBuilder.Build(observations, targets, indices, weights);
        }
    }
}
