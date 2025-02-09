using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.AdaBoost.Models;

/// <summary>
/// AdaBoost classification model. Consist of a series of tree model and corresponding weights
/// </summary>
[Serializable]
public sealed class ClassificationAdaBoostModel : IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
{
    readonly double[] m_modelWeights;
    readonly ClassificationDecisionTreeModel[] m_models;
    readonly double[] m_rawVariableImportance;
    readonly Dictionary<double, double> m_predictions = new();

    /// <summary>
    /// AdaBoost classification model. Consist of a series of tree model and corresponding weights
    /// </summary>
    /// <param name="models"></param>
    /// <param name="modelWeights"></param>
    /// <param name="rawVariableImportance"></param>
    public ClassificationAdaBoostModel(ClassificationDecisionTreeModel[] models, double[] modelWeights,
        double[] rawVariableImportance)
    {
        m_models = models ?? throw new ArgumentNullException(nameof(models));
        m_modelWeights = modelWeights ?? throw new ArgumentNullException(nameof(modelWeights));
        m_rawVariableImportance = rawVariableImportance ?? throw new ArgumentNullException(nameof(rawVariableImportance));
    }

    /// <summary>
    /// Predicts a single observations using weighted majority vote
    /// </summary>
    /// <param name="observation"></param>
    /// <returns></returns>
    public double Predict(double[] observation)
    {
        var count = m_models.Length;
        m_predictions.Clear();

        for (var i = 0; i < count; i++)
        {
            var prediction = m_models[i].Predict(observation);
            var weight = m_modelWeights[i];

            if (m_predictions.ContainsKey(prediction))
            {
                m_predictions[prediction] += weight;
            }
            else
            {
                m_predictions.Add(prediction, weight);
            }
        }

        return m_predictions.OrderByDescending(v => v.Value).First().Key;
    }

    /// <summary>
    /// Predicts a set of observations using weighted majority vote
    /// </summary>
    /// <param name="observations"></param>
    /// <returns></returns>
    public double[] Predict(F64Matrix observations)
    {
        var rows = observations.RowCount;
        var predictions = new double[rows];
        for (var i = 0; i < rows; i++)
        {
            predictions[i] = Predict(observations.Row(i));
        }

        return predictions;
    }

    /// <summary>
    /// Predicts a single observation using the ensembled probabilities
    /// Note this can yield a different result than using regular predict
    /// usually this will be a more accurate predictions
    /// </summary>
    /// <param name="observation"></param>
    /// <returns></returns>
    public ProbabilityPrediction PredictProbability(double[] observation)
    {
        var probabilities = new Dictionary<double, double>();
        var modelsProbability = m_models.Select(m => m.PredictProbability(observation).Probabilities)
            .ToArray();

        for (var i = 0; i < modelsProbability.Length; i++)
        {
            var model = modelsProbability[i];
            var w = m_modelWeights[i];

            foreach (var probability in model)
            {
                if (probabilities.ContainsKey(probability.Key))
                {
                    probabilities[probability.Key] += w * probability.Value;
                }
                else
                {
                    probabilities.Add(probability.Key, w * probability.Value);
                }
            }
        }

        var keys = probabilities.Keys.ToList();
        var probabilityFactor = 1.0 / m_modelWeights.Sum();

        foreach (var target in keys)
        {
            probabilities[target] *= probabilityFactor;
        }

        var prediction = probabilities.OrderByDescending(p => p.Value)
            .First().Key;

        return new ProbabilityPrediction(prediction, probabilities);
    }

    /// <summary>
    /// Predicts a set of observations using the ensembled probabilities
    /// Note this can yield a different result than using regular predict
    /// usually this will be a more accurate predictions
    /// </summary>
    /// <param name="observations"></param>
    /// <returns></returns>
    public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
    {
        var rows = observations.RowCount;
        var predictions = new ProbabilityPrediction[rows];
        for (var i = 0; i < rows; i++)
        {
            predictions[i] = PredictProbability(observations.Row(i));
        }

        return predictions;
    }

    /// <summary>
    /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
    /// </summary>
    /// <param name="featureNameToIndex"></param>
    /// <returns></returns>
    public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
    {
        var max = m_rawVariableImportance.Max();

        var scaledVariableImportance = m_rawVariableImportance
            .Select(v => (v / max) * 100.0)
            .ToArray();

        return featureNameToIndex.ToDictionary(kvp => kvp.Key, kvp => scaledVariableImportance[kvp.Value])
                    .OrderByDescending(kvp => kvp.Value)
                    .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
    }

    /// <summary>
    /// Gets the raw unsorted variable importance scores
    /// </summary>
    /// <returns></returns>
    public double[] GetRawVariableImportance() => m_rawVariableImportance;

    /// <summary>
    /// Loads a ClassificationAdaBoostModel.
    /// </summary>
    /// <param name="reader"></param>
    /// <returns></returns>
    public static ClassificationAdaBoostModel Load(Func<TextReader> reader)
    {
        return new GenericXmlDataContractSerializer()
            .Deserialize<ClassificationAdaBoostModel>(reader);
    }

    /// <summary>
    /// Saves the ClassificationAdaBoostModel.
    /// </summary>
    /// <param name="writer"></param>
    public void Save(Func<TextWriter> writer)
    {
        new GenericXmlDataContractSerializer()
            .Serialize(this, writer);
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
