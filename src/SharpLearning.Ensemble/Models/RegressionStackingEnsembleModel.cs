using System;
using System.Collections.Generic;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Ensemble.Models;

/// <summary>
/// Regression stacking ensemble model
/// </summary>
public class RegressionStackingEnsembleModel : IPredictorModel<double>
{
    readonly IPredictorModel<double> m_metaModel;
    readonly IPredictorModel<double>[] m_ensembleModels;
    readonly bool m_includeOriginalFeaturesForMetaLearner;

    /// <summary>
    ///
    /// </summary>
    /// <param name="ensembleModels">Models included in the ensemble</param>
    /// <param name="metaModel">Meta or top level model to combine the ensemble models</param>
    /// <param name="includeOriginalFeaturesForMetaLearner">True; the meta learner also receives the original features.
    /// False; the meta learner only receives the output of the ensemble models as features</param>
    public RegressionStackingEnsembleModel(IPredictorModel<double>[] ensembleModels, IPredictorModel<double> metaModel, bool includeOriginalFeaturesForMetaLearner)
    {
        m_ensembleModels = ensembleModels ?? throw new ArgumentNullException(nameof(ensembleModels));
        m_metaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
        m_includeOriginalFeaturesForMetaLearner = includeOriginalFeaturesForMetaLearner;
    }

    /// <summary>
    /// Predicts a single observation
    /// </summary>
    /// <param name="observation"></param>
    /// <returns></returns>
    public double Predict(double[] observation)
    {
        var ensembleCols = m_ensembleModels.Length;
        if (m_includeOriginalFeaturesForMetaLearner)
        {
            ensembleCols += observation.Length;
        }

        var ensemblePredictions = new double[ensembleCols];
        for (var i = 0; i < m_ensembleModels.Length; i++)
        {
            ensemblePredictions[i] = m_ensembleModels[i].Predict(observation);
        }

        if (m_includeOriginalFeaturesForMetaLearner)
        {
            Array.Copy(observation, 0, ensemblePredictions, m_ensembleModels.Length, observation.Length);
        }

        return m_metaModel.Predict(ensemblePredictions);
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
        for (var i = 0; i < observations.RowCount; i++)
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
    public double[] GetRawVariableImportance() => m_metaModel.GetRawVariableImportance();

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
                duplicateModelCount[name]++;
            }

            ensembleFeatureNameToIndex.Add(name + "_" + duplicateModelCount[name].ToString(), index++);
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
