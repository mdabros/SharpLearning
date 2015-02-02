using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Containers.Extensions;
using System;

namespace SharpLearning.GradientBoost.LossFunctions
{
    /// <summary>
    /// Least absolute deviation (LAD) loss function
    /// </summary>
    public sealed class LeastAbsoluteErrorLossFunction : ILossFunction
    {
        readonly double m_learningRate;
        double m_median;

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
        public LeastAbsoluteErrorLossFunction(double learningRate)
        {
            if (learningRate <= 0.0) { throw new ArgumentException("Learning rate must larger than 0.0"); }
            m_learningRate = learningRate;
        }

        /// <summary>
        /// Calculates the initial loss within the provided indices. The loss is stored in predictions.
        /// The initial loss is the median of the targets for LAD.
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
        /// Calculates the negative gradient between the targets and the prediction. 
        /// The gradient is returned in residuals. For LAD the negative gradient is 
        /// 1.0 if targets[i] - predictions[i] > 0.0 else -1.0
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        /// <param name="indices"></param>
        public void NegativeGradient(double[] targets, double[] predictions, double[] residuals, int[] indices)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                var value = targets[index] - predictions[index];
                if(value > 0.0)
                {
                    residuals[index] = 1.0;
                }
                else
                {
                    residuals[index] = -1.0;
                }
            }
        }

        /// <summary>
        /// Updates the tree model and predictions based on the targets and predictions. 
        /// LAD updates the value of the tree leafs with the median of (targets - predictions)
        /// for each leaf region. 
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
                if(tree.Nodes[i].FeatureIndex == -1)
                {
                    var node = tree.Nodes[i];
                    var nodeRegion = nodeIndices[node.LeafProbabilityIndex];

                    var diff = new double[nodeRegion.Count];
                    for (int j = 0; j < diff.Length; j++)
                    {
                        var index = nodeRegion[j];
                        diff[j] = targets[index] - predictions[index];
                    }

                    var newValue = diff.Median();

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
