using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System;
using System.Linq;

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

        public void InitializeLoss(double[] targets, double[] predictions)
        {
            m_mean = targets.Sum() / targets.Length;
            for (int i = 0; i < targets.Length; i++)
			{
                predictions[i] = m_mean;
			}
        }

        public void NegativeGradient(double[] targets, double[] predictions, double[] residuals)
        {
            for (int i = 0; i < targets.Length; i++)
            {
                residuals[i] = targets[i] - predictions[i];
            }
        }

        public void UpdateModel(BinaryTree tree, F64Matrix observations, double[] predictions)
        {
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] += m_learningRate * tree.Predict(observations.GetRow(i));
            }
        }
    }
}
