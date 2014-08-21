using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.GradientBoost.LossFunctions
{
    public interface ILossFunction
    {
        void InitializeLoss(double[] targets, double[] predictions, int[] indices);
        
        double LearningRate { get; }
        double InitialLoss { get; }

        void NegativeGradient(double[] targets, double[] predictions, double[] residuals, int[] indices);
        void UpdateModel(BinaryTree tree, F64Matrix observations, double[] predictions, int[] indices);
    }
}
