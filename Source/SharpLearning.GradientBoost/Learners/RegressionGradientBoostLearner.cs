using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.TreeBuilders;
using SharpLearning.GradientBoost.LossFunctions;
using SharpLearning.GradientBoost.Models;
using SharpLearning.Metrics.Regression;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpLearning.Containers;
using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.GradientBoost.Learners
{
    /// <summary>
    /// Regression gradient boost learner based on 
    /// http://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    /// A series of regression trees are fitted stage wise on the residuals of the previous stage.
    /// The resulting models are ensembled together using addition.
    /// </summary>
    public class RegressionGradientBoostLearner : IIndexedLearner<double>, ILearner<double>
    {
        readonly ILossFunction m_lossFunction;
        DecisionTreeLearner m_learner;
        readonly RandomIndexSampler<double> m_sampler = new RandomIndexSampler<double>(43);

        readonly int m_iterations;
        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;

        readonly double m_subSampleRatio;
        readonly int m_numberOfFeaturesPrSplit;

        int m_maximumTreeDepth;
        int m_maximumLeafCount;

        double[] m_redisuals = new double[0];
        double[] m_predictions = new double[0];

        List<RegressionDecisionTreeModel> m_models = new List<RegressionDecisionTreeModel>();

        /// <summary>
        ///  Regression gradient boost learner. 
        ///  A series of regression trees are fitted stage wise on the residuals of the previous stage
        /// </summary>
        /// <param name="lossFunction">The type of loss used calculating residuals</param>
        /// <param name="iterations">The number of iterations or stages</param>
        /// <param name="maximumTreeDepth">The maximum depth of the tree models</param>
        /// <param name="maximumLeafCount">The maximum leaf count of the tree models</param>
        /// <param name="minimumSplitSize">minimum node split size in the trees 1 is default</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">ratio of observations sampled at each iteration. Default is 1.0. 
        /// If below 1.0 the algorithm changes to stochastic gradient boosting. 
        /// This reduces variance in the ensemble and can help ounter overfitting</param>
        /// <param name="featuresPrSplit">Number of features used at each split in each tree. 0 means Sqrt(of availible features)</param>
        public RegressionGradientBoostLearner(ILossFunction lossFunction, int iterations, int maximumTreeDepth,
            int maximumLeafCount, int minimumSplitSize, double minimumInformationGain, double subSampleRatio, int numberOfFeaturesPrSplit)
        {
            if (lossFunction == null) { throw new ArgumentNullException("lossFunction"); } // currently only least squares is supported
            if (iterations < 1) { throw new ArgumentException("Iterations must be at least 1"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth < 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (maximumLeafCount <= 1) { throw new ArgumentException("maximum leaf count must be larger than 1"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }
            if (subSampleRatio <= 0.0 || subSampleRatio > 1.0) { throw new ArgumentException("subSampleRatio must be larger than 0.0 and at max 1.0"); }

            m_lossFunction = lossFunction;// currently only least squares is supported
            
            m_iterations = iterations;

            m_minimumSplitSize = minimumSplitSize;
            m_maximumTreeDepth = maximumTreeDepth;
            m_maximumLeafCount = maximumLeafCount;
            m_minimumInformationGain = minimumInformationGain;

            m_subSampleRatio = subSampleRatio;
            m_numberOfFeaturesPrSplit = numberOfFeaturesPrSplit;
        }

        /// <summary>
        /// Learns a RegressionGradientBoostModel 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public RegressionGradientBoostModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a RegressionGradientBoostModel
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public RegressionGradientBoostModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            m_learner = new DecisionTreeLearner(
                new BestFirstTreeBuilder(m_maximumTreeDepth, m_maximumLeafCount,
                    m_numberOfFeaturesPrSplit, m_minimumInformationGain, 42,
                    new OnlyUniqueThresholdsSplitSearcher(m_minimumSplitSize),
                    new RegressionImpurityCalculator()));

            m_models.Clear();

            Array.Clear(m_redisuals, 0, m_redisuals.Length);
            Array.Resize(ref m_redisuals, targets.Length);

            Array.Clear(m_predictions, 0, m_predictions.Length);
            Array.Resize(ref m_predictions, targets.Length);

            m_lossFunction.InitializeLoss(targets, m_predictions, indices);

            var sampleSize = (int)Math.Round(m_subSampleRatio * (double)indices.Length);


            for (int i = 0; i < m_iterations; i++)
            {
                if (m_subSampleRatio != 1.0)
                {
                    var trainingIndices = m_sampler.Sample(targets, sampleSize, indices);
                    FitStage(i, observations, targets, trainingIndices);
                }
                else
                {
                    FitStage(i, observations, targets, indices);
                }
            }

            var models = m_models.ToArray();
            var variableImportance = VariableImportance(models, observations.GetNumberOfColumns());

            return new RegressionGradientBoostModel(models, variableImportance,
                m_lossFunction.LearningRate, m_lossFunction.InitialLoss);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictor<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictor<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        void FitStage(int iteration, F64Matrix observations, double[] targets, int[] indices)
        {
            m_lossFunction.NegativeGradient(targets, m_predictions, m_redisuals, indices);

            var model = new RegressionDecisionTreeModel(m_learner.Learn(observations, m_redisuals, indices));

            m_lossFunction.UpdateModel(model.Tree, observations, targets, m_predictions, indices);

            m_models.Add(model);
        }

        double[] VariableImportance(RegressionDecisionTreeModel[] models, int numberOfFeatures)
        {
            var rawVariableImportance = new double[numberOfFeatures];

            foreach (var model in models)
            {
                var modelVariableImportance = model.GetRawVariableImportance();

                for (int j = 0; j < modelVariableImportance.Length; j++)
                {
                    rawVariableImportance[j] += modelVariableImportance[j];
                }
            }
            return rawVariableImportance;
        }
    }
}
