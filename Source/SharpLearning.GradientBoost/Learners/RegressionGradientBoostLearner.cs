using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.TreeBuilders;
using SharpLearning.GradientBoost.LossFunctions;
using SharpLearning.GradientBoost.Models;
using System;
using System.Collections.Generic;

namespace SharpLearning.GradientBoost.Learners
{
    public class RegressionGradientBoostLearner
    {
        readonly ILossFunction m_lossFunction;
        DecisionTreeLearner m_learner;

        readonly int m_iterations;
        readonly double m_learningRate;
        readonly int m_minimumSplitSize;
        readonly double m_minimumInformationGain;

        int m_maximumTreeDepth;
        int m_maximumLeafCount;

        double[] m_redisuals = new double[0];
        double[] m_predictions = new double[0];

        List<RegressionDecisionTreeModel> m_models = new List<RegressionDecisionTreeModel>();

        public RegressionGradientBoostLearner(int iterations = 100, double learningRate = 0.1, int maximumTreeDepth = 3, 
            int maximumLeafCount=2000, int minimumSplitSize = 1, double minimumInformationGain = 0.000001)
        {
            //if (lossFunction == null) { throw new ArgumentNullException("lossFunction"); } // currently only least squares is supported
            if (iterations < 1) { throw new ArgumentException("Iterations must be at least 1"); }
            if (learningRate > 1.0 || learningRate <= 0) { throw new ArgumentException("learningRate must be larger than zero and smaller than 1.0"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth < 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (maximumLeafCount <= 1) { throw new ArgumentException("maximum leaf count must be larger than 1"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }

            m_lossFunction = new LeastSquaresLossFunction(learningRate); // currently only least squares is supported
            
            m_iterations = iterations;
            m_learningRate = learningRate;

            m_minimumSplitSize = minimumSplitSize;
            m_maximumTreeDepth = maximumTreeDepth;
            m_maximumLeafCount = maximumLeafCount;
            m_minimumInformationGain = minimumInformationGain;
        }

        public RegressionGradientBoostModel Learn(F64Matrix observations, double[] targets)
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

            m_lossFunction.InitializeLoss(targets, m_predictions);

            for (int i = 0; i < m_iterations; i++)
            {
                FitStage(i, observations, targets);
            }

            var models = m_models.ToArray();
            var variableImportance = VariableImportance(models, observations.GetNumberOfColumns());
            var loss = new LeastSquaresLossFunction(m_learningRate, m_lossFunction.InitialLoss); // currently only least squares is supported

            return new RegressionGradientBoostModel(models, variableImportance, loss);
        }

        void FitStage(int iteration, F64Matrix observations, double[] targets)
        {
            m_lossFunction.NegativeGradient(targets, m_predictions, m_redisuals);
            
            var model = new RegressionDecisionTreeModel(m_learner.Learn(observations, m_redisuals));

            m_lossFunction.UpdateModel(model.Tree, observations, m_predictions);

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
