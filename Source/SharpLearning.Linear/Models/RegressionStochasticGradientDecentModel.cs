using SharpLearning.Containers.Matrices;
using SharpLearning.Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.Linear.Models
{
    /// <summary>
    /// Regression model learned using stochastic gradient descent
    /// </summary>
    [Serializable]
    public sealed class RegressionStochasticGradientDecentModel : IPredictorModel<double>
    {
        readonly double[] m_weights;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="weights">Weights foreach parameter including bias at index 0</param>
        public RegressionStochasticGradientDecentModel(double[] weights)
        {
            if (weights == null) { throw new ArgumentNullException("weights"); }
            m_weights = weights;
        }

        /// <summary>
        /// Predicts a single observation
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

            return prediction;
        }

        /// <summary>
        /// Predicts a set of observations
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var predictions = new double[observations.RowCount()];
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = Predict(observations.Row(i));
            }

            return predictions;
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name.
        /// Note that model importances RegressionStochasticGradientDecentModel are only valid 
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
        /// Note that model importances RegressionStochasticGradientDecentModel are only valid 
        /// if features have been scaled to equal range, like 0-1, before training 
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            return m_weights.Skip(1)
                .Select(w => Math.Abs(w)).ToArray();
        }

        /// <summary>
        /// Loads a RegressionStochasticGradientDecentModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static RegressionStochasticGradientDecentModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<RegressionStochasticGradientDecentModel>(reader);
        }

        /// <summary>
        /// Saves the RegressionStochasticGradientDecentModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }
    }
}
