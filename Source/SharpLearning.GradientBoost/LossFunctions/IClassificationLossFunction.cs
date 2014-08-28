using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System.Collections.Generic;

namespace SharpLearning.GradientBoost.LossFunctions
{
    public interface IClassificationLossFunction
    {
        /// <summary>
        /// Calculates the prior probabilities within the provided indices. The probabilities for each class is stored in 
        /// in a separate prediction array
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions">Loss is returned in predictions</param>
        /// <param name="indices"></param>
        void InitializePriorProbabilities(double[] targets, List<double[]> predictions, int[] indices);

        /// <summary>
        /// The learning rate of the loss function
        /// </summary>
        double LearningRate { get; }

        /// <summary>
        /// The constant value for initial loss
        /// </summary>
        double[] PriorProbabilities { get; }

        /// <summary>
        /// Calculates the negative gradient between the targets and the prediction. 
        /// The gradient is returned in residuals
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        /// <param name="indices"></param>
        void NegativeGradient(double[] targets, int targetIndex, List<double[]> predictions, double[] residuals, int[] indices);

        /// <summary>
        /// Updates the tree model and predictions based on the targets and predictions
        /// </summary>
        /// <param name="tree"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="indices"></param>
        void UpdateModel(BinaryTree tree, F64Matrix observations, double[] targets, double[] predictions, 
            double[] residuals, int[] indices);

    }
}
