using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Ensemble.Models
{
    /// <summary>
    /// Classification stacking ensemble model
    /// </summary>
    [Serializable]
    public class ClassificationStackingEnsembleModel : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        readonly IPredictorModel<ProbabilityPrediction> m_metaModel;
        readonly IPredictorModel<ProbabilityPrediction>[] m_ensembleModels;
        readonly bool m_includeOriginalFeaturesForMetaLearner;
        readonly int m_numberOfClasses;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ensembleModels">Models included in the ensemble</param>
        /// <param name="metaModel">Meta or top level model to combine the ensemble models</param>
        /// <param name="includeOriginalFeaturesForMetaLearner">True; the meta learner also receives the original features. 
        /// False; the meta learner only receives the output of the ensemble models as features</param>
        /// <param name="numberOfClasses">Number of classes in the classification problem</param>
        public ClassificationStackingEnsembleModel(IPredictorModel<ProbabilityPrediction>[] ensembleModels,
            IPredictorModel<ProbabilityPrediction> metaModel,
            bool includeOriginalFeaturesForMetaLearner,
            int numberOfClasses)
        {
            m_ensembleModels = ensembleModels ?? throw new ArgumentNullException(nameof(ensembleModels));
            m_metaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
            m_includeOriginalFeaturesForMetaLearner = includeOriginalFeaturesForMetaLearner;
            m_numberOfClasses = numberOfClasses;
        }

        /// <summary>
        /// Predicts a single observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return PredictProbability(observation).Prediction;
        }

        /// <summary>
        /// Predicts a set of observations
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
        /// Predicts a single observation using the ensembled probabilities
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            var ensembleFeatures = m_ensembleModels.Length * m_numberOfClasses;
            var ensembleCols = ensembleFeatures;
            if (m_includeOriginalFeaturesForMetaLearner)
            {
                ensembleCols = ensembleCols + observation.Length;
            }

            var ensemblePredictions = new double[ensembleCols];
            for (int i = 0; i < m_ensembleModels.Length; i++)
            {
                var probabilities = m_ensembleModels[i].Predict(observation)
                    .Probabilities.Values.ToArray();
                for (int j = 0; j < m_numberOfClasses; j++)
                {
                    ensemblePredictions[i * m_numberOfClasses + j] = probabilities[j];
                }
            }

            if (m_includeOriginalFeaturesForMetaLearner)
            {
                Array.Copy(observation, 0, ensemblePredictions, ensembleFeatures, observation.Length);
            }

            return m_metaModel.Predict(ensemblePredictions);
        }

        /// <summary>
        /// Predicts a set of observations using the ensembled probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var predictions = new ProbabilityPrediction[observations.RowCount];
            var observation = new double[observations.ColumnCount];
            for (int i = 0; i < observations.RowCount; i++)
            {
                observations.Row(i, observation);
                predictions[i] = PredictProbability(observation);
            }

            return predictions;
        }

        /// <summary>
        /// Gets the raw unsorted variable importance scores
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            return m_metaModel.GetRawVariableImportance();
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var ensembleFeatureNameToIndex = new Dictionary<string, int>();
            var index = 0;

            var duplicateModelCount = new Dictionary<string, int>();

            foreach (var model in m_ensembleModels)
            {
                var name = model.GetType().Name;

                if (!duplicateModelCount.ContainsKey(name))
                {
                    duplicateModelCount.Add(name, 0);
                }
                else
                {
                    duplicateModelCount[name] += 1;
                }

                name += "_" + duplicateModelCount[name].ToString();

                for (int j = 0; j < m_numberOfClasses; j++)
                {
                    ensembleFeatureNameToIndex.Add(name + "_Class_Probability_" + j, index++);
                }
            }

            if (m_includeOriginalFeaturesForMetaLearner)
            {
                foreach (var feature in featureNameToIndex)
                {
                    var name = GetNewFeatureName(feature.Key, ensembleFeatureNameToIndex);
                    ensembleFeatureNameToIndex.Add(name, index++);
                }
            }

            return m_metaModel.GetVariableImportance(ensembleFeatureNameToIndex);
        }

        /// <summary>
        /// Private explicit interface implementation for probability predictions
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        ProbabilityPrediction IPredictor<ProbabilityPrediction>.Predict(double[] observation)
            => PredictProbability(observation);

        /// <summary>
        /// Private explicit interface implementation for probability predictions
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        ProbabilityPrediction[] IPredictor<ProbabilityPrediction>.Predict(F64Matrix observations)
            => PredictProbability(observations);

        static string GetNewFeatureName(string name, Dictionary<string, int> ensembleFeatureNameToIndex)
        {
            if (ensembleFeatureNameToIndex.ContainsKey(name))
            {
                name += "_PreviousStack";
                return GetNewFeatureName(name, ensembleFeatureNameToIndex);
            }
            else
            {
                return name;
            }
        }
    }
}
