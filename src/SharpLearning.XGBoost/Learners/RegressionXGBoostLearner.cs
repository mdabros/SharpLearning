using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.XGBoost.Models;
using XGBoost.lib;

namespace SharpLearning.XGBoost.Learners
{
    /// <summary>
    /// Regression leaner for XGBoost
    /// </summary>
    public sealed class RegressionXGBoostLearner : ILearner<double>
    {
        readonly IDictionary<string, object> m_parameters = new Dictionary<string, object>();

        /// <summary>
        /// 
        /// </summary>
        /// <param name="maxDepth"></param>
        /// <param name="learningRate"></param>
        /// <param name="estimaters"></param>
        /// <param name="silent"></param>
        /// <param name="objective"></param>
        /// <param name="boosterType"></param>
        /// <param name="treeMethod"></param>
        /// <param name="numberOfThreads"></param>
        /// <param name="gamma"></param>
        /// <param name="minChildWeight"></param>
        /// <param name="maxDeltaStep"></param>
        /// <param name="subsample"></param>
        /// <param name="colSampleByTree"></param>
        /// <param name="colSampleByLevel"></param>
        /// <param name="regAlpha"></param>
        /// <param name="regLambda"></param>
        /// <param name="scalePosWeight"></param>
        /// <param name="baseScore"></param>
        /// <param name="seed"></param>
        /// <param name="missing"></param>
        public RegressionXGBoostLearner(int maxDepth = 3, double learningRate = 0.1F, int estimaters = 100,
            bool silent = true, 
            RegressionObjective objective = RegressionObjective.Linear,
            BoosterType boosterType = BoosterType.GBTree,
            TreeMethod treeMethod = TreeMethod.Auto,
            int numberOfThreads = -1, double gamma = 0, int minChildWeight = 1,
            int maxDeltaStep = 0, double subsample = 1, double colSampleByTree = 1,
            double colSampleByLevel = 1, double regAlpha = 0, double regLambda = 1,
            double scalePosWeight = 1, double baseScore = 0.5F, int seed = 0,
            double missing = double.NaN)
        {
            m_parameters[ParameterNames.MaxDepth] = maxDepth;
            m_parameters[ParameterNames.LearningRate] = (float)learningRate;
            m_parameters[ParameterNames.Estimators] = estimaters;
            m_parameters[ParameterNames.Silent] = silent;
            m_parameters[ParameterNames.objective] = objective.ToXGBoostString();

            m_parameters[ParameterNames.Threads] = numberOfThreads;
            m_parameters[ParameterNames.Gamma] = (float)gamma;
            m_parameters[ParameterNames.MinChildWeight] = minChildWeight;
            m_parameters[ParameterNames.MaxDeltaStep] = maxDeltaStep;
            m_parameters[ParameterNames.SubSample] = (float)subsample;
            m_parameters[ParameterNames.ColSampleByTree] = (float)colSampleByTree;
            m_parameters[ParameterNames.ColSampleByLevel] = (float)colSampleByLevel;
            m_parameters[ParameterNames.RegAlpha] = (float)regAlpha;
            m_parameters[ParameterNames.RegLambda] = (float)regLambda;
            m_parameters[ParameterNames.ScalePosWeight] = (float)scalePosWeight;

            m_parameters[ParameterNames.BaseScore] = (float)baseScore;
            m_parameters[ParameterNames.Seed] = seed;
            m_parameters[ParameterNames.Missing] = (float)missing;
            m_parameters[ParameterNames.ExistingBooster] = null;
            m_parameters[ParameterNames.Booster] = boosterType.ToXGBoostString();
            m_parameters[ParameterNames.TreeMethod] = treeMethod.ToXGBoostString();
        }

        /// <summary>
        /// Learns an XGBoost regression model.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public RegressionXGBoostModel Learn(F64Matrix observations, double[] targets)
        {
            var floatObservations = observations.ToFloatJaggedArray();
            var floatTargets = targets.ToFloat();

            using (var train = new DMatrix(floatObservations, floatTargets))
            {
                var booster = new Booster(m_parameters.ToDictionary(v => v.Key, v => v.Value), train);
                var iterations = (int)m_parameters[ParameterNames.Estimators];

                for (int iteration = 0; iteration < iterations; iteration++)
                {
                    booster.Update(train, iteration);
                }
                
                return new RegressionXGBoostModel(booster);
            }
        }

        /// <summary>
        /// Learns an XGBoost regression model.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }
    }
}
