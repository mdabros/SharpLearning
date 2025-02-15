using System;
using System.Diagnostics;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Ensemble.Models;
using SharpLearning.Ensemble.Strategies;

namespace SharpLearning.Ensemble.Learners;

/// <summary>
/// Regression ensemble learner.
/// http://mlwave.com/kaggle-ensembling-guide/
/// </summary>
public sealed class RegressionEnsembleLearner : ILearner<double>, IIndexedLearner<double>
{
    readonly Func<F64Matrix, double[], int[], IPredictorModel<double>>[] m_learners;
    readonly Func<IRegressionEnsembleStrategy> m_ensembleStrategy;
    readonly double m_subSampleRatio;
    readonly Random m_random;

    /// <summary>
    /// Regression ensemble learner. Combines several models into a single ensemble model.
    /// Default combination method is the mean of all model outputs.
    /// </summary>
    /// <param name="learners">Learners in the ensemble</param>
    /// <param name="subSampleRatio">Default is 1.0. All models are trained on all data. 
    /// If different from 1.0 models are trained using bagging with the chosen sub sample ratio</param>
    /// <param name="seed">Seed for the bagging when used</param>
    public RegressionEnsembleLearner(
        IIndexedLearner<double>[] learners,
        double subSampleRatio = 1.0,
        int seed = 24)
        : this(learners.Select(l => new Func<F64Matrix, double[], int[], IPredictorModel<double>>((o, t, i) => l.Learn(o, t, i))).ToArray(),
            () => new MeanRegressionEnsembleStrategy(), subSampleRatio, seed)
    {
    }

    /// <summary>
    /// Regression ensemble learner. Combines several models into a single ensemble model.
    /// </summary>
    /// <param name="learners">Learners in the ensemble</param>
    /// <param name="ensembleStrategy">Strategy on how to combine the models. Default is mean of all models in the ensemble</param>
    /// <param name="subSampleRatio">Default is 1.0. All models are trained on all data. 
    /// If different from 1.0 models are trained using bagging with the chosen sub sample ratio</param>
    /// <param name="seed">Seed for the bagging when used</param>
    public RegressionEnsembleLearner(
        IIndexedLearner<double>[] learners,
        IRegressionEnsembleStrategy ensembleStrategy,
        double subSampleRatio = 1.0,
        int seed = 24)
        : this(learners.Select(l => new Func<F64Matrix, double[], int[], IPredictorModel<double>>((o, t, i) => l.Learn(o, t, i))).ToArray(),
            () => ensembleStrategy, subSampleRatio, seed)
    {
    }

    /// <summary>
    /// Regression ensemble learner. Combines several models into a single ensemble model.
    /// </summary>
    /// <param name="learners">Learners in the ensemble</param>
    /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
    /// <param name="subSampleRatio">Default is 1.0. All models are trained on all data. 
    /// If different from 1.0 models are trained using bagging with the chosen sub sample ratio</param>
    /// <param name="seed">Seed for the bagging when used</param>
    public RegressionEnsembleLearner(
        Func<F64Matrix, double[], int[], IPredictorModel<double>>[] learners,
        Func<IRegressionEnsembleStrategy> ensembleStrategy,
        double subSampleRatio = 1.0,
        int seed = 24)
    {
        m_learners = learners ?? throw new ArgumentNullException(nameof(learners));
        if (learners.Length < 1) { throw new ArgumentException("there must be at least 1 learner"); }
        m_ensembleStrategy = ensembleStrategy ?? throw new ArgumentNullException(nameof(ensembleStrategy));

        m_random = new Random(seed);
        m_subSampleRatio = subSampleRatio;
    }

    /// <summary>
    /// Learns a regression ensemble
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    public RegressionEnsembleModel Learn(F64Matrix observations, double[] targets)
    {
        var indices = Enumerable.Range(0, targets.Length).ToArray();
        return Learn(observations, targets, indices);
    }

    /// <summary>
    /// Learns a regression ensemble on the provided indices
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <param name="indices"></param>
    /// <returns></returns>
    public RegressionEnsembleModel Learn(F64Matrix observations, double[] targets,
        int[] indices)
    {
        Checks.VerifyObservationsAndTargets(observations, targets);
        Checks.VerifyIndices(indices, observations, targets);

        var ensembleModels = new IPredictorModel<double>[m_learners.Length];
        var sampleSize = (int)Math.Round(m_subSampleRatio * indices.Length);

        if (sampleSize < 1) { throw new ArgumentException("subSampleRatio two small"); }

        var inSample = new int[sampleSize];

        for (var i = 0; i < m_learners.Length; i++)
        {
            Trace.WriteLine("Training model: " + (i + 1));

            if (m_subSampleRatio != 1.0)
            {
                Sample(inSample, indices);
                ensembleModels[i] = m_learners[i](observations, targets, inSample);
            }
            else
            {
                ensembleModels[i] = m_learners[i](observations, targets, indices);
            }
        }

        return new RegressionEnsembleModel(ensembleModels, m_ensembleStrategy());
    }


    void Sample(int[] inSample, int[] allIndices)
    {
        for (var i = 0; i < inSample.Length; i++)
        {
            var index = m_random.Next(0, allIndices.Length - 1);
            inSample[i] = allIndices[index];
        }
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
    /// Private explicit interface implementation for learning.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="targets"></param>
    /// <returns></returns>
    IPredictorModel<double> ILearner<double>.Learn(
        F64Matrix observations, double[] targets) => Learn(observations, targets);
}
