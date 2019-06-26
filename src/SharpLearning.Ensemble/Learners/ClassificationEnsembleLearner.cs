﻿using System;
using System.Diagnostics;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Ensemble.Models;
using SharpLearning.Ensemble.Strategies;

namespace SharpLearning.Ensemble.Learners
{
    /// <summary>
    /// Classification ensemble learner.
    /// http://mlwave.com/kaggle-ensembling-guide/
    /// </summary>
    public sealed class ClassificationEnsembleLearner 
        : ILearner<ProbabilityPrediction>
        , IIndexedLearner<ProbabilityPrediction>
        , ILearner<double>
        , IIndexedLearner<double>
    {
        readonly Func<F64Matrix, double[], int[], IPredictorModel<ProbabilityPrediction>>[] m_learners;
        readonly Func<IClassificationEnsembleStrategy> m_ensembleStrategy;
        readonly double m_subSampleRatio;
        readonly Random m_random;

        /// <summary>
        /// Classification ensemble learner. Combines several models into a single ensemble model.
        /// Default combination method is mean of the probabilities of the models.
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="subSampleRatio">Default is 1.0. All models are trained on all data. 
        /// If different from 1.0 models are trained using bagging with the chosen sub sample ratio</param>
        /// <param name="seed">Seed for the bagging when used</param>
        public ClassificationEnsembleLearner(
            IIndexedLearner<ProbabilityPrediction>[] learners, 
            double subSampleRatio = 1.0, 
            int seed = 24)
            : this(learners.Select(l => new Func<F64Matrix, double[], int[], IPredictorModel<ProbabilityPrediction>>((o, t, i) => l.Learn(o, t, i))).ToArray(),
                () => new MeanProbabilityClassificationEnsembleStrategy(), subSampleRatio, seed)
        {
        }

        /// <summary>
        /// Classification ensemble learner. Combines several models into a single ensemble model.
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
        /// <param name="subSampleRatio">Default is 1.0. All models are trained on all data. 
        /// If different from 1.0 models are trained using bagging with the chosen sub sample ratio</param>
        /// <param name="seed">Seed for the bagging when used</param>
        public ClassificationEnsembleLearner(
            IIndexedLearner<ProbabilityPrediction>[] learners, 
            IClassificationEnsembleStrategy ensembleStrategy,
            double subSampleRatio = 1.0, 
            int seed = 24)
            : this(learners.Select(l => new Func<F64Matrix, double[], int[], IPredictorModel<ProbabilityPrediction>>((o, t, i) => l.Learn(o, t, i))).ToArray(), 
                () => ensembleStrategy, subSampleRatio, seed)
        {
        }

        /// <summary>
        /// Classification ensemble learner. Combines several models into a single ensemble model.
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="ensembleStrategy">Strategy on how to combine the models</param>
        /// <param name="subSampleRatio">Default is 1.0. All models are trained on all data. 
        /// If different from 1.0 models are trained using bagging with the chosen sub sample ratio</param>
        /// <param name="seed">Seed for the bagging when used</param>
        public ClassificationEnsembleLearner(
            Func<F64Matrix, double[], int[], IPredictorModel<ProbabilityPrediction>>[] learners, 
            Func<IClassificationEnsembleStrategy> ensembleStrategy,
            double subSampleRatio = 1.0, 
            int seed = 24)
        {
            m_learners = learners ?? throw new ArgumentNullException(nameof(learners));
            if (learners.Length < 1) { throw new ArgumentException("there must be at least 1 learner"); }
            m_ensembleStrategy = ensembleStrategy ?? throw new ArgumentNullException(nameof(ensembleStrategy));

            m_learners = learners;
            m_random = new Random(seed);
            m_subSampleRatio = subSampleRatio;
        }

        /// <summary>
        /// Learns a classification ensemble
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationEnsembleModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a classification ensemble on the provided indices
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ClassificationEnsembleModel Learn(F64Matrix observations, double[] targets, 
            int[] indices)
        {
            var ensembleModels = new IPredictorModel<ProbabilityPrediction>[m_learners.Length];
            var sampleSize = (int)Math.Round(m_subSampleRatio * indices.Length);

            if(sampleSize < 1) { throw new ArgumentException("subSampleRatio two small"); }

            var inSample = new int[sampleSize];

            for (int i = 0; i < m_learners.Length; i++)
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

            return new ClassificationEnsembleModel(ensembleModels, m_ensembleStrategy());
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

        /// <summary>
        /// Random sampling
        /// </summary>
        /// <param name="inSample"></param>
        /// <param name="allIndices"></param>
        void Sample(int[] inSample, int[] allIndices)
        {
            for (int i = 0; i < inSample.Length; i++)
            {
                var index = m_random.Next(0, allIndices.Length - 1);
                inSample[i] = allIndices[index];
            }
        }
    }
}
