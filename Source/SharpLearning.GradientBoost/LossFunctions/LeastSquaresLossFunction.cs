using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System;

namespace SharpLearning.GradientBoost.LossFunctions
{
    /// <summary>
    /// Least squares (LS) loss function
    /// </summary>
    public sealed class LeastSquaresLossFunction : ILossFunction
    {
        readonly double m_learningRate;
        double m_mean;

        /// <summary>
        /// The learning rate of the loss function
        /// </summary>
        public double LearningRate { get { return m_learningRate; } }

        /// <summary>
        /// The initial loss (mean of the training targets)
        /// </summary>
        public double InitialLoss { get { return m_mean; } }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate"></param>
        public LeastSquaresLossFunction(double learningRate)
        {
            if (learningRate <= 0.0) { throw new ArgumentException("Learning rate must larger than 0.0"); }
            m_learningRate = learningRate;
        }

        /// <summary>
        /// Calculates the initial loss within the provided indices. The loss is stored in predictions.
        /// The initial loss is the mean of the targets for LS.
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="indices"></param>
        public void InitializeLoss(double[] targets, double[] predictions, int[] indices)
        {
            m_mean = targets.Sum(indices) / indices.Length;
            for (int i = 0; i < indices.Length; i++)
			{
                var index = indices[i];
                predictions[index] = m_mean;
			}
        }

        /// <summary>
        /// Calculates the negative gradient between the targets and the prediction. 
        /// The gradient is returned in residuals. For LS the negative gradient is 
        /// targets[i] - predictions[i]
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
                residuals[index] = targets[index] - predictions[index];
            }
        }

        /// <summary>
        /// Updates the tree model and predictions based on the targets and predictions. 
        /// LS does not update the value of the tree leafs. Only the predictions are updated
        /// </summary>
        /// <param name="tree"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="indices"></param>
        public void UpdateModel(BinaryTree tree, F64Matrix observations, double[] targets, double[] predictions, int[] indices)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                predictions[index] += m_learningRate * tree.Predict(observations.GetRow(index));
            }
        }
    }
}
