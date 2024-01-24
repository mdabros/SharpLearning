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
using SharpLearning.Metrics.Regression;

namespace SharpLearning.AdaBoost.Learners
{
    /// <summary>
    /// Regression AdaBoost learner using the R2 algorithm 
    /// using weighted sampling to target the observations with largest error and
    /// weighted median to ensemble the models.
    /// </summary>
    public sealed class RegressionAdaBoostLearner : IIndexedLearner<double>, ILearner<double>
    {
        readonly int m_iterations;
        readonly double m_learningRate;

        readonly int m_minimumSplitSize;
        int m_maximumTreeDepth;
        readonly double m_minimumInformationGain;

        RegressionDecisionTreeLearner m_modelLearner;

        readonly MeanAbsolutErrorRegressionMetric m_errorMetric = new MeanAbsolutErrorRegressionMetric();
        readonly AdaBoostRegressionLoss m_loss;

        List<double> m_modelErrors = new List<double>();
        List<double> m_modelWeights = new List<double>();
        List<RegressionDecisionTreeModel> m_models = new List<RegressionDecisionTreeModel>();

        double[] m_workErrors = new double[0];
        double[] m_sampleWeights = new double[0];
        double[] m_indexedTargets = new double[0];
        int[] m_sampleIndices = new int[0];

        readonly WeightedRandomSampler m_sampler;

        /// <summary>
        /// Regression AdaBoost learner using the R2 algorithm 
        /// using weighted sampling to target the observations with largest error and
        /// weighted median to ensemble the models.
        /// </summary>
        /// <param name="iterations">Number of iterations (models) to boost</param>
        /// <param name="learningRate">How much each boost iteration should add (between 1.0 and 0.0)</param>
        /// <param name="maximumTreeDepth">The maximum depth of the tree models. 
        /// 0 will set the depth to default 3</param>
        /// <param name="loss">Type of loss used when boosting weights. Linear is default</param>
        /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="seed">Seed for the random sampling</param>
        public RegressionAdaBoostLearner(int iterations = 50,
            double learningRate = 1,
            int maximumTreeDepth = 0,
            AdaBoostRegressionLoss loss = AdaBoostRegressionLoss.Linear,
            int minimumSplitSize = 1,
            double minimumInformationGain = 0.000001,
            int seed = 42)
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
            m_loss = loss;
            m_minimumInformationGain = minimumInformationGain;

            m_sampler = new WeightedRandomSampler(seed);
        }

        /// <summary>
        /// Learns an Adaboost regression model
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public RegressionAdaBoostModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns an Adaboost regression model
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public RegressionAdaBoostModel Learn(F64Matrix observations, double[] targets,
            int[] indices)
        {
            Checks.VerifyObservationsAndTargets(observations, targets);
            Checks.VerifyIndices(indices, observations, targets);

            if (m_maximumTreeDepth == 0)
            {
                m_maximumTreeDepth = 3;
            }

            m_modelLearner = new RegressionDecisionTreeLearner(m_maximumTreeDepth, m_minimumSplitSize,
                observations.ColumnCount, m_minimumInformationGain, 42);

            m_modelErrors.Clear();
            m_modelWeights.Clear();
            m_models.Clear();

            Array.Resize(ref m_sampleWeights, targets.Length);

            Array.Resize(ref m_workErrors, targets.Length);
            Array.Resize(ref m_indexedTargets, indices.Length);
            Array.Resize(ref m_sampleIndices, indices.Length);

            indices.IndexedCopy(targets, Interval1D.Create(0, indices.Length),
                m_indexedTargets);

            var initialWeight = 1.0 / indices.Length;
            for (int i = 0; i < indices.Length; i++)
            {
                var index = indices[i];
                m_sampleWeights[index] = initialWeight;
            }

            for (int i = 0; i < m_iterations; i++)
            {
                if (!Boost(observations, targets, indices, i))
                    break;

                var ensembleError = ErrorEstimate(observations, indices);

                if (ensembleError == 0.0)
                    break;

                if (m_modelErrors[i] == 0.0)
                    break;

                var weightSum = m_sampleWeights.Sum(indices);
                if (weightSum <= 0.0)
                    break;

                if (i == m_iterations - 1)
                {
                    // Normalize weights
                    for (int j = 0; j < indices.Length; j++)
                    {
                        var index = indices[j];
                        m_sampleWeights[index] = m_sampleWeights[index] / weightSum;
                    }
                }
            }

            var featuresCount = observations.ColumnCount;
            var variableImportance = VariableImportance(featuresCount);

            return new RegressionAdaBoostModel(m_models.ToArray(), m_modelWeights.ToArray(),
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
        /// Private explicit interface implementation for learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(
            F64Matrix observations, double[] targets) => Learn(observations, targets);

        bool Boost(F64Matrix observations, double[] targets, int[] indices, int iteration)
        {
            m_sampler.Sample(indices, m_sampleWeights, m_sampleIndices);

            var model = m_modelLearner.Learn(observations, targets,
                m_sampleIndices); // weighted sampling is used instead of weights in training


            var predictions = model.Predict(observations, indices);

            for (int i = 0; i < predictions.Length; i++)
            {
                var index = indices[i];
                m_workErrors[index] = Math.Abs(m_indexedTargets[i] - predictions[i]);
            }

            var maxError = m_workErrors.Max();

            for (int i = 0; i < m_workErrors.Length; i++)
            {

                var error = m_workErrors[i];

                if (maxError != 0.0)
                {
                    error = error / maxError;
                }

                switch (m_loss)
                {
                    case AdaBoostRegressionLoss.Linear:
                        break;
                    case AdaBoostRegressionLoss.Squared:
                        error = error * error;
                        break;
                    case AdaBoostRegressionLoss.Exponential:
                        error = 1.0 - Math.Exp(-error);
                        break;
                    default:
                        throw new ArgumentException("Unsupported loss type");
                }

                m_workErrors[i] = error;
            }

            var modelError = m_workErrors.WeightedMean(m_sampleWeights, indices);

            if (modelError <= 0.0)
            {
                m_modelErrors.Add(0.0);
                m_modelWeights.Add(1.0);
                m_models.Add(model);
                return true;
            }
            else if (modelError >= 0.5)
            {
                return false;
            }

            var beta = modelError / (1.0 - modelError);

            var modelWeight = m_learningRate * Math.Log(1.0 / beta);

            // Only boost if not last iteration
            if (iteration != m_iterations - 1)
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    var index = indices[i];
                    var sampleWeight = m_sampleWeights[index];
                    var error = m_workErrors[index];
                    m_sampleWeights[index] = sampleWeight * Math.Pow(beta, (1.0 - error) * m_learningRate);
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

            for (int i = 0; i < rows; i++)
            {
                var index = indices[i];
                predictions[i] = Predict(observations.Row(index));
            }

            var error = m_errorMetric.Error(m_indexedTargets, predictions);

            //Trace.WriteLine("Error: " + error);

            return error;
        }

        double Predict(double[] observation)
        {
            var count = m_models.Count;
            var predictions = new double[count];

            for (int i = 0; i < count; i++)
            {
                predictions[i] = m_models[i].Predict(observation);
            }

            var weights = m_modelWeights.ToArray();
            predictions.SortWith(weights);

            var prediction = predictions.WeightedMedian(weights);

            return prediction;
        }

        double[] VariableImportance(int featuresCount)
        {
            var variableImportance = new double[featuresCount];
            for (int i = 0; i < m_models.Count; i++)
            {
                var w = m_modelWeights[i];
                var modelImportances = m_models[i].GetRawVariableImportance();

                for (int j = 0; j < featuresCount; j++)
                {
                    variableImportance[j] += w * modelImportances[j];
                }
            }
            return variableImportance;
        }
    }
}
