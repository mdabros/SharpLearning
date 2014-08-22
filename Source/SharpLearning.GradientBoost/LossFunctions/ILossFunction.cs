using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.GradientBoost.LossFunctions
{
    /// <summary>
    /// Interface for GradientBoosting loss functions
    /// </summary>
    public interface ILossFunction
    {
        /// <summary>
        /// Calculates the initial loss within the provided indices. The loss is stored in predictions
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions">Loss is returned in predictions</param>
        /// <param name="indices"></param>
        void InitializeLoss(double[] targets, double[] predictions, int[] indices);
        
        /// <summary>
        /// The learning rate of the loss function
        /// </summary>
        double LearningRate { get; }

        /// <summary>
        /// The constant value for initial loss
        /// </summary>
        double InitialLoss { get; }

        /// <summary>
        /// Calculates the negative gradient between the targets and the prediction. 
        /// The gradient is returned in residuals
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        /// <param name="indices"></param>
        void NegativeGradient(double[] targets, double[] predictions, double[] residuals, int[] indices);
        
        /// <summary>
        /// Updates the tree model and predictions based on the targets and predictions
        /// </summary>
        /// <param name="tree"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="indices"></param>
        void UpdateModel(BinaryTree tree, F64Matrix observations, double[] targets, double[] predictions, int[] indices);
    }
}
