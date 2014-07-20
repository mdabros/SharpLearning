using SharpLearning.AdaBoost.Models;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.Metrics.Classification;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.AdaBoost.Learning
{
    /// <summary>
    /// Classification AdaBoost learner using the SAMME algorithm for multi-class support:
    /// http://web.stanford.edu/~hastie/Papers/samme.pdf
    /// </summary>
    public sealed class ClassificationAdaBoostLearner
    {
        readonly int m_iterations;
        readonly double m_learningRate;

        readonly int m_minimumSplitSize;
        readonly int m_maximumTreeDepth;
        readonly double m_minimumInformationGain;

        int m_uniqueTargetValues;
        DecisionTreeLearner m_modelLearner;

        readonly TotalErrorClassificationMetric<double> m_errorMetric = new TotalErrorClassificationMetric<double>();

        List<double> m_modelErrors = new List<double>();
        List<double> m_modelWeights = new List<double>();
        List<ClassificationDecisionTreeModel> m_models = new List<ClassificationDecisionTreeModel>();
        
        double[] m_workErrors = new double[0];
        double[] m_sampleWeights = new double[0];

        /// <summary>
        /// Classification AdaBoost learner using the SAMME algorithm for multi-class support:
        /// http://web.stanford.edu/~hastie/Papers/samme.pdf
        /// </summary>
        /// <param name="iterations">Number of iterations (models) to boost</param>
        /// <param name="learningRate">How much each boost iteration should add (between 1.0 and 0.0)</param>
        /// <param name="maximumTreeDepth">The maximum depth of the tree models. 
        /// for 2 class problem 1 is usually enough. For more classes or larger problems between 3 to 8 is recommended</param>
        /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        public ClassificationAdaBoostLearner(int iterations = 50, double learningRate = 1, int maximumTreeDepth = 3, int minimumSplitSize = 1, double minimumInformationGain = 0.000001)
        {
            if (iterations < 1) { throw new ArgumentException("Iterations must be at least 1"); }
            if (learningRate > 1.0 || learningRate <= 0) { throw new ArgumentException("learningRate must be larger than zero and smaller than 1.0"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
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
            m_modelLearner = new DecisionTreeLearner(m_maximumTreeDepth, observations.GetNumberOfColumns(),
                m_minimumInformationGain, 42,
                new OnlyUniqueThresholdsSplitSearcher(m_minimumSplitSize),
                new GiniClasificationImpurityCalculator());

            m_uniqueTargetValues = targets.Distinct().Count();

            m_modelErrors.Clear();
            m_modelWeights.Clear();
            m_models.Clear();

            Array.Resize(ref m_sampleWeights, targets.Length);
            Array.Resize(ref m_workErrors, targets.Length);


            var initialWeight = 1.0 / targets.Length;
            for (int i = 0; i < m_sampleWeights.Length; i++)
            {
                m_sampleWeights[i] = initialWeight;
            }

            for (int i = 0; i < m_iterations; i++)
            {
                if (!Boost(observations, targets, i))
                    break;

                var ensembleError = ErrorEstimate(observations, targets);

                if (ensembleError == 0.0)
                    break;

                if (m_modelErrors[i] == 0.0)
                    break;

                var weightSum = m_sampleWeights.Sum();
                if (weightSum <= 0.0)
                    break;
                
                if (i == m_iterations - 1)
                {
                    // Normalize weights
                    for (int j = 0; j < m_sampleWeights.Length; j++)
                    {
                        m_sampleWeights[j] = m_sampleWeights[j] / weightSum;
                    }
                }
            }

            var featuresCount = observations.GetNumberOfColumns();
            var variableImportance = VariableImportance(featuresCount);

            return new ClassificationAdaBoostModel(m_models.ToArray(), m_modelWeights.ToArray(), 
                variableImportance);
        }

        bool Boost(F64Matrix observations, double[] targets, int iteration)
        {
            var model = new ClassificationDecisionTreeModel(m_modelLearner.Learn(observations, targets, m_sampleWeights),
                m_modelLearner.m_variableImportance);

            var predictions = model.Predict(observations);

            for (int i = 0; i < predictions.Length; i++)
            {
                if (targets[i] != predictions[i])
                {
                    m_workErrors[i] = 1.0;
                }
                else
                {
                    m_workErrors[i] = 0.0;
                }
            }

            var modelError = m_workErrors.WeightedMean(m_sampleWeights);

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
                for (int i = 0; i < m_sampleWeights.Length; i++)
                {
                    var sampleWeight = m_sampleWeights[i];
                    if(sampleWeight > 0.0 || modelWeight < 0.0)
                    {
                        m_sampleWeights[i] = sampleWeight * Math.Exp(modelWeight * m_workErrors[i]);
                    }
                }
            }

            m_modelErrors.Add(modelError);
            m_modelWeights.Add(modelWeight);
            m_models.Add(model);

            return true;
        }

        double ErrorEstimate(F64Matrix observations, double[] targets)
        {
            var rows = targets.Length;
            var predictions = new double[rows];
            
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = Predict(observations.GetRow(i));
            }

            var error = m_errorMetric.Error(targets, predictions);

            //Trace.WriteLine("Error: " + error);
            //Trace.WriteLine(m_errorMetric.ErrorString(targets, predictions));

            return error;
        }

        double Predict(double[] observation)
        {
            var count = m_models.Count;
            var predictions = new Dictionary<double, double>();

            for (int i = 0; i < count; i++)
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
