using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using System;
using System.Collections.Generic;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class GBMGradientBoostClassificationModel
    {
        readonly GBMTree[] m_trees;
        readonly double m_learningRate;

        public GBMGradientBoostClassificationModel(GBMTree[] trees, double learningRate)
        {
            if (trees == null) { throw new ArgumentNullException("trees"); }
            m_trees = trees;
            m_learningRate = learningRate;
        }

        /// <summary>
        /// Predicts a single observations using the combination of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var probability = Probability(observation);
            var prediction = (probability >= 0.5) ? 1.0 : 0.0;
            return prediction;
        }

        double Probability(double[] observation)
        {
            var prediction = m_trees[0].Predict(observation);
            for (int i = 1; i < m_trees.Length; i++)
            {
                prediction += m_learningRate * m_trees[i].Predict(observation);
            }

            return Sigmoid(prediction);
        }

        double Sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        /// <summary>
        /// Predicts a single observation with probabilities
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var probability = Probability(observation);
            var prediction = (probability >= 0.5) ? 1.0 : 0.0;
            var probabilities = new Dictionary<double, double> { { 0.0, 1.0 - probability }, { 1.0, probability } };

            return new ProbabilityPrediction(prediction, probabilities);
        }


        /// <summary>
        /// Predicts a set of obervations using the combination of all predictors
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = Predict(observations.GetRow(i));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts a set of observations with probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = PredictProbability(observations.GetRow(i));
            }

            return predictions;
        }
    }
}
