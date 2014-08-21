using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System;

namespace SharpLearning.GradientBoost.LossFunctions
{
    public sealed class LeastSquaresLossFunction : ILossFunction
    {
        readonly double m_learningRate;
        double m_mean;

        public double LearningRate { get { return m_learningRate; } }
        public double InitialLoss { get { return m_mean; } }

        public LeastSquaresLossFunction(double learningRate)
        {
            if (learningRate <= 0.0) { throw new ArgumentException("Learning rate must larger than 0.0"); }
            m_learningRate = learningRate;
        }

        public LeastSquaresLossFunction(double learningRate, double initialLoss)
        {
            if (learningRate <= 0.0) { throw new ArgumentException("Learning rate must larger than 0.0"); }
            m_learningRate = learningRate;
            m_mean = initialLoss;
        }

        public void InitializeLoss(double[] targets, double[] predictions, int[] indices)
        {
            m_mean = targets.Sum(indices) / indices.Length;
            for (int i = 0; i < indices.Length; i++)
			{
                var index = indices[i];
                predictions[index] = m_mean;
			}
        }

        public void NegativeGradient(double[] targets, double[] predictions, double[] residuals, int[] indices)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                residuals[index] = targets[index] - predictions[index];
            }
        }

        public void UpdateModel(BinaryTree tree, F64Matrix observations, double[] predictions, int[] indices)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                predictions[index] += m_learningRate * tree.Predict(observations.GetRow(index));
            }
        }
    }
}
