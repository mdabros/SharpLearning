using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.Ensemble.Strategies;

namespace SharpLearning.Ensemble.Models
{
    /// <summary>
    /// Regression ensemble model
    /// </summary>
    [Serializable]
    public class RegressionEnsembleModel : IPredictorModel<double>
    {
        readonly IRegressionEnsembleStrategy m_ensembleStrategy;
        readonly IPredictorModel<double>[] m_ensembleModels;

        /// <summary>
        /// Regression ensemble model
        /// </summary>
        /// <param name="ensembleModels">Models included in the ensemble</param>
        /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
        public RegressionEnsembleModel(IPredictorModel<double>[] ensembleModels, 
            IRegressionEnsembleStrategy ensembleStrategy)
        {
            m_ensembleModels = ensembleModels ?? throw new ArgumentNullException(nameof(ensembleModels));
            m_ensembleStrategy = ensembleStrategy ?? throw new ArgumentNullException(nameof(ensembleStrategy));
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var ensembleCols = m_ensembleModels.Length;

            var ensemblePredictions = new double[ensembleCols];
            for (int i = 0; i < m_ensembleModels.Length; i++)
            {
                ensemblePredictions[i] = m_ensembleModels[i].Predict(observation);
            }

            return m_ensembleStrategy.Combine(ensemblePredictions);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var predictions = new double[observations.RowCount];
            var observation = new double[observations.ColumnCount];
            for (int i = 0; i < observations.RowCount; i++)
            {
                observations.Row(i, observation);
                predictions[i] = Predict(observation);
            }

            return predictions;
        }
        
        /// <summary>
        /// Gets the raw unsorted variable importance scores
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            // return normalized variable importance. 
            // Individual models can have very different scaling of importances 
            var index = 0;
            var dummyFeatureNameToIndex = m_ensembleModels
                .First().GetRawVariableImportance()
                .ToDictionary(k => index.ToString(), k => index++);

            return GetVariableImportance(dummyFeatureNameToIndex).Values.ToArray();
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name 
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var variableImportance = featureNameToIndex.ToDictionary(k => k.Key, v => 0.0);

            foreach (var model in m_ensembleModels)
            {
                var modelImportances = model.GetVariableImportance(featureNameToIndex);
                foreach (var importance in modelImportances)
                {
                    variableImportance[importance.Key] += importance.Value;                    
                }
            }

            var max = variableImportance.Values.Max();

            return variableImportance
                 .OrderByDescending(kvp => kvp.Value)
                 .ToDictionary(k => k.Key, v => (v.Value / max) * 100.0);
        }
    }
}
