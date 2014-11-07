using SharpLearning.Containers.Matrices;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Linear.Models
{
    /// <summary>
    /// Linear regression model learned using stochastic gradient descent
    /// </summary>
    public sealed class ClassificationStochasticGradientDecentModel
    {
        readonly double[] m_weights;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="weights">Weights foreach parameter including bias at index 0</param>
        public ClassificationStochasticGradientDecentModel(double[] weights)
        {
            if (weights == null) { throw new ArgumentNullException("weights"); }
            m_weights = weights;
        }

        /// <summary>
        /// Predicts a single observation using the linear model
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = 1.0 * m_weights[0]; // bias
            for (int i = 0; i < observation.Length; i++)
            {
                prediction += m_weights[i + 1] * observation[i];
            }

            var probabilty = Sigmoid(prediction);

            if (probabilty >= 0.5)
            {
                return 1.0;
            }
            else
            {
                return 0.0;
            }
        }

        double Sigmoid(double z)
        {
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        /// <summary>
        /// Predicts a set of observations using the linear model
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var predictions = new double[observations.GetNumberOfRows()];
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = Predict(observations.GetRow(i));
            }

            return predictions;
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name.
        /// Note that model importances ClassificationStochasticGradientDecentModel are only valid 
        /// if features have been scaled to equal range, like 0-1, before training 
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var importances = m_weights
                .Select(w => Math.Abs(w)).ToArray();

            var max = importances.Max();

            var scaledVariableImportance = importances
                .Select(v => (v / max) * 100.0)
                .ToArray();

            return featureNameToIndex.ToDictionary(kvp => kvp.Key, kvp => scaledVariableImportance[kvp.Value])
                        .OrderByDescending(kvp => kvp.Value)
                        .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }

        /// <summary>
        /// Gets the raw unsorted variable importance scores.
        /// Note that model importances ClassificationStochasticGradientDecentModel are only valid 
        /// if features have been scaled to equal range, like 0-1, before training 
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            return m_weights.Select(w => Math.Abs(w)).ToArray();
        }
    }
}
