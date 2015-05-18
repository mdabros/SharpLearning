using SharpLearning.Containers.Matrices;
using System;

namespace SharpLearning.GradientBoost.GBM
{
    /// <summary>
    /// 
    /// </summary>
    public sealed class GBMGradientBoostRegressorModel
    {
        readonly GBMTree[] m_trees;
        readonly double m_learningRate;

        public GBMGradientBoostRegressorModel(GBMTree[] trees, double learningRate)
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
            var prediction = m_trees[0].Predict(observation);
            for (int i = 1; i < m_trees.Length; i++)
            {
                prediction += m_learningRate * m_trees[i].Predict(observation);
            }

            return prediction;
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
    }
}
