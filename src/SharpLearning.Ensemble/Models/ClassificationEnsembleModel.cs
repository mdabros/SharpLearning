using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Ensemble.Strategies;

namespace SharpLearning.Ensemble.Models;

/// <summary>
/// Classification ensemble model
/// </summary>
[Serializable]
public class ClassificationEnsembleModel : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
{
    readonly IClassificationEnsembleStrategy m_ensembleStrategy;
    readonly IPredictorModel<ProbabilityPrediction>[] m_ensembleModels;

    /// <summary>
    /// Classification ensemble model
    /// </summary>
    /// <param name="ensembleModels">Models included in the ensemble</param>
    /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
    public ClassificationEnsembleModel(IPredictorModel<ProbabilityPrediction>[] ensembleModels,
        IClassificationEnsembleStrategy ensembleStrategy)
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
        return PredictProbability(observation).Prediction;
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
        for (var i = 0; i < observations.RowCount; i++)
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
        var ensembleCols = m_ensembleModels.Length;

        var ensemblePredictions = new ProbabilityPrediction[ensembleCols];
        for (var i = 0; i < m_ensembleModels.Length; i++)
        {
            ensemblePredictions[i] = m_ensembleModels[i].Predict(observation);
        }

        return m_ensembleStrategy.Combine(ensemblePredictions);
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
        for (var i = 0; i < observations.RowCount; i++)
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
        // return normalized variable importance. 
        // Individual models can have very different scaling of importances 
        var index = 0;
        var dummyFeatureNameToIndex = m_ensembleModels[0].GetRawVariableImportance()
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
}
