using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Learners.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Linear.Models
{
    /// <summary>
    /// Linear regression model learned using stochastic gradient descent
    /// </summary>
    public sealed class ClassificationStochasticGradientDecentModel : IPredictor<double>, IPredictor<ProbabilityPrediction>
    {
        readonly Dictionary<double, BinaryClassificationStochasticGradientDecentModel> m_models;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="weights">Weights foreach parameter including bias at index 0</param>
        public ClassificationStochasticGradientDecentModel(Dictionary<double, BinaryClassificationStochasticGradientDecentModel> models)
        {
            if (models == null) { throw new ArgumentNullException("models"); }
            m_models = models;
        }

        /// <summary>
        /// Predicts a single observation using the linear model
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var props = m_models.Select(m => new { Target = m.Key, Probability = m.Value.PredictProbability(observation).Probabilities[1.0] })
                .OrderByDescending(m => m.Probability).ToArray();

            return props.First().Target;
        }

        /// <summary>
        /// Private explicit interface implementation for probability predictions
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        ProbabilityPrediction IPredictor<ProbabilityPrediction>.Predict(double[] observation)
        {
            return PredictProbability(observation);
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
        /// Predicts a single observation with probabilities
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var probabilities = m_models.Select(m => new { Target = m.Key, Probability = 
                m.Value.PredictProbability(observation).Probabilities[1.0] })
                .ToDictionary(p => p.Target, p => p.Probability);

            var prediction = probabilities.OrderByDescending(p => p.Value)
                .First().Key;

            return new ProbabilityPrediction(prediction, probabilities);
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

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name.
        /// Note that model importances ClassificationStochasticGradientDecentModel are only valid 
        /// if features have been scaled to equal range, like 0-1, before training 
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var importances = GetRawVariableImportance();

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
            var importances = new double[m_models.First().Value.GetRawVariableImportance().Length];
            foreach (var model in m_models.Values)
            {
                var modelImportances = model.GetRawVariableImportance();
                for (int i = 0; i < importances.Length; i++)
                {
                    if(importances[i] < modelImportances[i])
                    {
                        importances[i] = modelImportances[i];
                    }
                }
            }

            return importances;
        }
    }
}
