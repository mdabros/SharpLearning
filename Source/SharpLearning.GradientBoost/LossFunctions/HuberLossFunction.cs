using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System;
using System.Linq;

namespace SharpLearning.GradientBoost.LossFunctions
{
    /// <summary>
    /// Huber loss function
    /// </summary>
    public sealed class HuberLossFunction : ILossFunction
    {
        readonly double m_learningRate;
        readonly double m_alpha;
        double m_median;
        double m_gamma;

        /// <summary>
        /// The learning rate of the loss function
        /// </summary>
        public double LearningRate { get { return m_learningRate; } }

        /// <summary>
        /// The initial loss (median of the training targets)
        /// </summary>
        public double InitialLoss { get { return m_median; } }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="alpha"></param>
        public HuberLossFunction(double learningRate, double alpha=0.9)
        {
            if (learningRate <= 0.0) { throw new ArgumentException("Learning rate must larger than 0.0"); }
            if (alpha <= 0.0 || alpha > 1.0) { throw new ArgumentException("Learning rate must larger than 0.0"); }
            m_learningRate = learningRate;
            m_alpha = alpha;
        }

        /// <summary>
        /// Calculates the initial loss within the provided indices. The loss is stored in predictions.
        /// The initial loss is the median of the targets for Huber.
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="indices"></param>
        public void InitializeLoss(double[] targets, double[] predictions, int[] indices)
        {
            m_median = targets.Median(indices);

            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                predictions[index] = m_median;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        /// <param name="indices"></param>
        public void NegativeGradient(double[] targets, double[] predictions, double[] residuals, int[] indices)
        {
            var absDiff = new double[indices.Length];
            var difference = new double[indices.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                var value = targets[index] - predictions[index];
                difference[i] = value;
                absDiff[i] = Math.Abs(value);
            }

            var gamma = absDiff.ScoreAtPercentile(m_alpha);

            for (int i = 0; i < indices.Length; i++)
            {
                var diff = absDiff[i];
                var index = indices[i];

                if(diff <= gamma)
                {
                    residuals[index] = difference[i];
                }
                else
                {

                    residuals[index] = gamma * Math.Sign(difference[i]);
                }
            }

            m_gamma = gamma;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tree"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="indices"></param>
        public void UpdateModel(BinaryTree tree, F64Matrix observations, double[] targets, double[] predictions, int[] indices)
        {
            var nodeIndices = tree.LeafRegionIndices(observations, indices);

            for (int i = 0; i < tree.Nodes.Count; i++)
            {
                if (tree.Nodes[i].FeatureIndex == -1)
                {
                    var node = tree.Nodes[i];
                    var nodeRegion = nodeIndices[node.LeafProbabilityIndex];

                    var diff = new double[nodeRegion.Count];
                    for (int j = 0; j < diff.Length; j++)
                    {
                        var index = nodeRegion[j];
                        diff[j] = targets[index] - predictions[index];
                    }

                    var median = diff.Median();
                    var values = new double[diff.Length];

                    for (int j = 0; j < diff.Length; j++)
                    {
                        var index = nodeRegion[j];
                        var medianDiff = diff[j] - median;
                        var sign = Math.Sign(medianDiff);

                        values[j] = sign * Math.Min(Math.Abs(medianDiff), m_gamma); 
                    }

                    var newValue = median + values.Sum() / values.Length;

                    tree.Nodes[i] = new Node(node.FeatureIndex, newValue, node.LeftIndex,
                        node.RightIndex, node.NodeIndex, node.LeafProbabilityIndex);
                }
            }

            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                predictions[index] += m_learningRate * tree.Predict(observations.GetRow(index));
            }
        }
    }
}
