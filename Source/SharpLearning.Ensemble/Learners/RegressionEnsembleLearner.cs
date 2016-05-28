using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.Ensemble.Models;
using SharpLearning.Ensemble.Strategies;
using System;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.Ensemble.Learners
{
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
        public RegressionEnsembleLearner(IIndexedLearner<double>[] learners, double subSampleRatio = 1.0, int seed = 24)
            : this(learners.Select(l => new Func<F64Matrix, double[], int[], IPredictorModel<double>>((o, t, i) => l.Learn(o, t, i))).ToArray(),
            () => new MeanRegressionEnsembleStrategy(), subSampleRatio, seed)
        {
        }

        /// <summary>
        /// Regression ensemble learner. Combines several models into a single ensemble model.
        /// </summary>
        /// <param name="learners">Learners in the ensemble</param>
        /// <param name="ensembleStrategy">Strategy on how to combine the models. Default is mean of all models in the ensmble</param>
        /// <param name="subSampleRatio">Default is 1.0. All models are trained on all data. 
        /// If different from 1.0 models are trained using bagging with the chosen sub sample ratio</param>
        /// <param name="seed">Seed for the bagging when used</param>
        public RegressionEnsembleLearner(IIndexedLearner<double>[] learners, IRegressionEnsembleStrategy ensembleStrategy,
            double subSampleRatio = 1.0, int seed = 24)
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
        public RegressionEnsembleLearner(Func<F64Matrix, double[], int[], IPredictorModel<double>>[] learners, Func<IRegressionEnsembleStrategy> ensembleStrategy,
            double subSampleRatio = 1.0, int seed = 24)
        {
            if (learners == null) { throw new ArgumentNullException("learners"); }
            if (ensembleStrategy == null) { throw new ArgumentNullException("ensembleStrategy"); }
            if (learners.Length < 1) { throw new ArgumentException("there must be at least 1 learner"); }
            m_learners = learners;
            m_ensembleStrategy = ensembleStrategy;
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
        public RegressionEnsembleModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            var ensembleModels = new IPredictorModel<double>[m_learners.Length];
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

            return new RegressionEnsembleModel(ensembleModels, m_ensembleStrategy());
        }


        void Sample(int[] inSample, int[] allIndices)
        {
            for (int i = 0; i < inSample.Length; i++)
            {
                var index = m_random.Next(0, allIndices.Length - 1);
                inSample[i] = allIndices[index];
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }
    }
}
