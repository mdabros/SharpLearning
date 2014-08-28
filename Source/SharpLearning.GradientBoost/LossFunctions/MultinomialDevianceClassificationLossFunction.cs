using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.GradientBoost.LossFunctions
{
    /// <summary>
    /// Multinomial deviance loss function for multi-class classification
    /// </summary>
    public sealed class MultinomialDevianceClassificationLossFunction : IClassificationLossFunction
    {
        readonly double m_learningRate;
        double[] m_priorProbabilities = new double[0];

        /// <summary>
        /// The learning rate of the loss function
        /// </summary>
        public double LearningRate
        {
            get { return m_learningRate; }
        }

        /// <summary>
        /// The prior probabilities
        /// </summary>
        public double[] PriorProbabilities
        {
            get { return m_priorProbabilities; }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="learningRate"></param>
        public MultinomialDevianceClassificationLossFunction(double learningRate)
        {
            if (learningRate <= 0.0) { throw new ArgumentException("Learning rate must larger than 0.0"); }
            m_learningRate = learningRate;
        }
        
        /// <summary>
        /// Initialize the prior probabilities for each class
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="indices"></param>
        public void InitializePriorProbabilities(double[] targets, List<double[]> predictions, int[] indices)
        {
            var targetCounts = new Dictionary<double, double>();
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                var target = targets[index];
                if(!targetCounts.ContainsKey(target))
                {
                    targetCounts.Add(target, 1);
                }
                else
                {
                    targetCounts[target]++;
                }
            }

            m_priorProbabilities = targetCounts.OrderBy(kvp => kvp.Key)
                .Select(kvp => (double)kvp.Value / (double)indices.Length)
                .ToArray();

            for (int j = 0; j < predictions.Count; j++)
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    var index = indices[i];
                    predictions[j][index] = m_priorProbabilities[j];
		        }
            }
        }

        /// <summary>
        /// Calculates the negative gradient using the residuals for each probability estimate
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="targetIndex"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        /// <param name="indices"></param>
        public void NegativeGradient(double[] targets, int targetIndex, List<double[]> predictions, double[] residuals, int[] indices)
        {
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                var target = targets[index];
                var prediction = predictions[targetIndex][index];
                var expPredictions = predictions.Exp(index);

                residuals[index] = target - Math.Exp(prediction - Math.Log(expPredictions.Sum()))
                            .NanToNum(); 
            }
        }

        /// <summary>
        /// Updates the tree model and predictions
        /// </summary>
        /// <param name="tree"></param>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="predictions"></param>
        /// <param name="residuals"></param>
        /// <param name="indices"></param>
        public void UpdateModel(BinaryTree tree, F64Matrix observations, double[] targets, double[] predictions, double[] residuals, int[] indices)
        {
            var nodeIndices = tree.LeafRegionIndices(observations, indices);

            for (int i = 0; i < tree.Nodes.Count; i++)
            {
                if (tree.Nodes[i].FeatureIndex == -1)
                {
                    var node = tree.Nodes[i];
                    var nodeRegion = nodeIndices[node.LeafProbabilityIndex];

                    var nodeResiduals = new double[nodeRegion.Count];
                    var denominators = new double[nodeRegion.Count];

                    for (int j = 0; j < nodeRegion.Count; j++)
                    {
                        var index = nodeRegion[j];
                        var residual = residuals[index];
                        var target = targets[index];
                        nodeResiduals[j] = residual;
                        denominators[j] = (target - residual) * (1.0 - target + residual);
                    }

                    var numerator = nodeResiduals.Sum();
                    numerator *= ((double)m_priorProbabilities.Length - 1.0) / (double)m_priorProbabilities.Length;

                    var denominator = denominators.Sum();
                    var newValue = 0.0;

                    if (denominator != 0.0)
                        newValue = numerator / denominator;

                    tree.Nodes[i] = new Node(node.FeatureIndex, newValue, node.LeftIndex,
                        node.RightIndex, node.NodeIndex, node.LeafProbabilityIndex);
                }

                for (int j = 0; j < indices.Length; j++)
                {
                    var index = indices[j];
                    predictions[index] += m_learningRate * tree.Predict(observations.GetRow(index));
                }
            }
        }
    }
}
