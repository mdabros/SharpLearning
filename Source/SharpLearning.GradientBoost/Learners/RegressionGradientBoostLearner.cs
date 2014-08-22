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

namespace SharpLearning.GradientBoost.Learners
{
    /// <summary>
    /// Regression gradient boost learner based on 
    /// http://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    /// A series of regression trees are fitted stage wise on the residuals of the previous stage.
    /// The resulting models are ensembled together using addition.
    /// </summary>
    public class RegressionGradientBoostLearner
    {
        readonly ILossFunction m_lossFunction;
        DecisionTreeLearner m_learner;

        readonly int m_iterations;
        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;

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
        public RegressionGradientBoostLearner(ILossFunction lossFunction, int iterations, int maximumTreeDepth, 
            int maximumLeafCount, int minimumSplitSize, double minimumInformationGain)
        {
            if (lossFunction == null) { throw new ArgumentNullException("lossFunction"); } // currently only least squares is supported
            if (iterations < 1) { throw new ArgumentException("Iterations must be at least 1"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth < 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (maximumLeafCount <= 1) { throw new ArgumentException("maximum leaf count must be larger than 1"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }

            m_lossFunction = lossFunction;// currently only least squares is supported
            
            m_iterations = iterations;

            m_minimumSplitSize = minimumSplitSize;
            m_maximumTreeDepth = maximumTreeDepth;
            m_maximumLeafCount = maximumLeafCount;
            m_minimumInformationGain = minimumInformationGain;
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
                    observations.GetNumberOfColumns(), m_minimumInformationGain, 42,
                    new OnlyUniqueThresholdsSplitSearcher(m_minimumSplitSize),
                    new RegressionImpurityCalculator()));

            m_models.Clear();

            Array.Clear(m_redisuals, 0, m_redisuals.Length);
            Array.Resize(ref m_redisuals, targets.Length);

            Array.Clear(m_predictions, 0, m_predictions.Length);
            Array.Resize(ref m_predictions, targets.Length);

            m_lossFunction.InitializeLoss(targets, m_predictions, indices);
            var evaluator = new MeanAbsolutErrorRegressionMetric();
            for (int i = 0; i < m_iterations; i++)
            {
                FitStage(i, observations, targets, indices);

                //Trace.WriteLine(evaluator.Error(targets.GetIndices(indices),
                //    m_predictions.GetIndices(indices)));
            }

            var models = m_models.ToArray();
            var variableImportance = VariableImportance(models, observations.GetNumberOfColumns());

            return new RegressionGradientBoostModel(models, variableImportance,
                m_lossFunction.LearningRate, m_lossFunction.InitialLoss);
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
