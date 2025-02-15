using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.AdaBoost.Models;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.AdaBoost.Learners;

/// <summary>
/// Classification AdaBoost learner using the SAMME algorithm for multi-class support:
/// http://web.stanford.edu/~hastie/Papers/samme.pdf
/// </summary>
public sealed class ClassificationAdaBoostLearner
    : IIndexedLearner<double>
    , IIndexedLearner<ProbabilityPrediction>
    , ILearner<double>
    , ILearner<ProbabilityPrediction>
{
    readonly int m_iterations;
    readonly double m_learningRate;

    readonly int m_minimumSplitSize;
    int m_maximumTreeDepth;
    readonly double m_minimumInformationGain;

    int m_uniqueTargetValues;
    ClassificationDecisionTreeLearner m_modelLearner;

    readonly TotalErrorClassificationMetric<double> m_errorMetric = new();

    readonly List<double> m_modelErrors = [];
    readonly List<double> m_modelWeights = [];
    readonly List<ClassificationDecisionTreeModel> m_models = [];

    double[] m_workErrors = [];
    double[] m_sampleWeights = [];
    double[] m_indexedTargets = [];

    /// <summary>
    /// Classification AdaBoost learner using the SAMME algorithm for multi-class support:
    /// http://web.stanford.edu/~hastie/Papers/samme.pdf
    /// </summary>
    /// <param name="iterations">Number of iterations (models) to boost</param>
    /// <param name="learningRate">How much each boost iteration should add (between 1.0 and 0.0)</param>
    /// <param name="maximumTreeDepth">The maximum depth of the tree models.
    /// for 2 class problem 1 is usually enough. For more classes or larger problems between 3 to 8 is recommended.
    /// 0 will set the depth equal to the number of classes in the problem</param>
    /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
    /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
    public ClassificationAdaBoostLearner(int iterations = 50,
        double learningRate = 1,
        int maximumTreeDepth = 0,
        int minimumSplitSize = 1,
        double minimumInformationGain = 0.000001)
    {
        if (iterations < 1) { throw new ArgumentException("Iterations must be at least 1"); }
        if (learningRate > 1.0 || learningRate <= 0) { throw new ArgumentException("learningRate must be larger than zero and smaller than 1.0"); }
        if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
        if (maximumTreeDepth < 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
        if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }

        m_iterations = iterations;
        m_learningRate = learningRate;

        m_minimumSplitSize = minimumSplitSize;
        m_maximumTreeDepth = maximumTreeDepth;
        m_minimumInformationGain = minimumInformationGain;
    }

    /// <summary>
    /// Learn an adaboost classification model
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    public ClassificationAdaBoostModel Learn(F64Matrix observations, double[] targets)
    {
        var indices = Enumerable.Range(0, targets.Length).ToArray();
        return Learn(observations, targets, indices);
    }

    /// <summary>
    /// Learn an adaboost classification model
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    public ClassificationAdaBoostModel Learn(F64Matrix observations, double[] targets,
        int[] indices)
    {
        Checks.VerifyObservationsAndTargets(observations, targets);
        Checks.VerifyIndices(indices, observations, targets);

        var uniques = new HashSet<double>();

        for (var i = 0; i < indices.Length; i++)
        {
            var value = targets[indices[i]];
            uniques.Add(value);
        }

        m_uniqueTargetValues = uniques.Count;

        if (m_maximumTreeDepth == 0)
        {
            m_maximumTreeDepth = m_uniqueTargetValues;
        }

        m_modelLearner = new ClassificationDecisionTreeLearner(m_maximumTreeDepth, m_minimumSplitSize,
            observations.ColumnCount, m_minimumInformationGain, 42);

        m_modelErrors.Clear();
        m_modelWeights.Clear();
        m_models.Clear();

        Array.Resize(ref m_sampleWeights, targets.Length);

        Array.Resize(ref m_workErrors, targets.Length);
        Array.Resize(ref m_indexedTargets, indices.Length);

        indices.IndexedCopy(targets, Interval1D.Create(0, indices.Length),
            m_indexedTargets);

        var initialWeight = 1.0 / indices.Length;
        for (var i = 0; i < indices.Length; i++)
        {
            var index = indices[i];
            m_sampleWeights[index] = initialWeight;
        }

        for (var i = 0; i < m_iterations; i++)
        {
            if (!Boost(observations, targets, indices, i))
            {
                break;
            }

            var ensembleError = ErrorEstimate(observations, indices);

            if (ensembleError == 0.0)
            {
                break;
            }

            if (m_modelErrors[i] == 0.0)
            {
                break;
            }

            var weightSum = m_sampleWeights.Sum(indices);
            if (weightSum <= 0.0)
            {
                break;
            }

            if (i == m_iterations - 1)
            {
                // Normalize weights
                for (var j = 0; j < indices.Length; j++)
                {
                    var index = indices[j];
                    m_sampleWeights[index] /= weightSum;
                }
            }
        }

        var featuresCount = observations.ColumnCount;
        var variableImportance = VariableImportance(featuresCount);

        return new ClassificationAdaBoostModel(m_models.ToArray(), m_modelWeights.ToArray(),
            variableImportance);
    }

    /// <summary>
    /// Private explicit interface implementation for indexed learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    IPredictorModel<double> IIndexedLearner<double>.Learn(
        F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

    /// <summary>
    /// Private explicit interface implementation for indexed probability learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(
        F64Matrix observations, double[] targets, int[] indices) => Learn(observations, targets, indices);

    /// <summary>
    /// Private explicit interface implementation.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    IPredictorModel<double> ILearner<double>.Learn(
        F64Matrix observations, double[] targets) => Learn(observations, targets);

    /// <summary>
    /// Private explicit interface implementation for probability learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(
        F64Matrix observations, double[] targets) => Learn(observations, targets);

    bool Boost(F64Matrix observations, double[] targets, int[] indices, int iteration)
    {
        var model = m_modelLearner.Learn(observations, targets,
            indices, m_sampleWeights);

        var predictions = model.Predict(observations, indices);

        for (var i = 0; i < predictions.Length; i++)
        {
            var index = indices[i];
            if (m_indexedTargets[i] != predictions[i])
            {
                m_workErrors[index] = 1.0;
            }
            else
            {
                m_workErrors[index] = 0.0;
            }
        }

        var modelError = m_workErrors.WeightedMean(m_sampleWeights, indices);

        if (modelError <= 0.0)
        {
            m_modelErrors.Add(0.0);
            m_modelWeights.Add(1.0);
            m_models.Add(model);
            return true;
        }

        var errorThreshold = 1.0 - (1.0 / (double)m_uniqueTargetValues);

        if (modelError >= errorThreshold)
        {
            return false;
        }

        var modelWeight = m_learningRate * (
            Math.Log((1.0 - modelError) / modelError) +
            Math.Log(m_uniqueTargetValues - 1.0));

        // Only boost if not last iteration
        if (iteration != m_iterations - 1)
        {
            for (var i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                var sampleWeight = m_sampleWeights[index];
                if (sampleWeight > 0.0 || modelWeight < 0.0)
                {
                    m_sampleWeights[index] = sampleWeight * Math.Exp(modelWeight * m_workErrors[index]);
                }
            }
        }

        m_modelErrors.Add(modelError);
        m_modelWeights.Add(modelWeight);
        m_models.Add(model);

        return true;
    }

    double ErrorEstimate(F64Matrix observations, int[] indices)
    {
        var rows = indices.Length;
        var predictions = new double[rows];

        for (var i = 0; i < rows; i++)
        {
            var index = indices[i];
            predictions[i] = Predict(observations.Row(index));
        }

        var error = m_errorMetric.Error(m_indexedTargets, predictions);

        //Trace.WriteLine("Error: " + error);
        //Trace.WriteLine(m_errorMetric.ErrorString(m_indexedTargets, predictions));

        return error;
    }

    double Predict(double[] observation)
    {
        var count = m_models.Count;
        var predictions = new Dictionary<double, double>();

        for (var i = 0; i < count; i++)
        {
            var prediction = m_models[i].Predict(observation);
            var weight = m_modelWeights[i];

            if (predictions.ContainsKey(prediction))
            {
                predictions[prediction] += weight;
            }
            else
            {
                predictions.Add(prediction, weight);
            }
        }

        return predictions.OrderByDescending(v => v.Value).First().Key;
    }

    double[] VariableImportance(int featuresCount)
    {
        var variableImportance = new double[featuresCount];
        for (var i = 0; i < m_models.Count; i++)
        {
            var w = m_modelWeights[i];
            var modelImportances = m_models[i].GetRawVariableImportance();

            for (var j = 0; j < featuresCount; j++)
            {
                variableImportance[j] += w * modelImportances[j];
            }
        }
        return variableImportance;
    }
}
