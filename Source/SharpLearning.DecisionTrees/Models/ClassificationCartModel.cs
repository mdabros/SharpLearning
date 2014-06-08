using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using System;

namespace SharpLearning.DecisionTrees.Models
{
    /// <summary>
    /// CART Decision tree model
    /// </summary>
    public sealed class ClassificationCartModel
    {
        readonly IBinaryDecisionNode m_root;

        public ClassificationCartModel(IBinaryDecisionNode root)
        {
            if (root == null) { throw new ArgumentNullException("root"); }
            m_root = root;
        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return m_root.Predict(observation);
        }

        /// <summary>
        /// Predicts a set of observations 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.GetNumberOfRows();
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = m_root.Predict(observations.GetRow(i));
            }

            return predictions;
        }
    }
}
