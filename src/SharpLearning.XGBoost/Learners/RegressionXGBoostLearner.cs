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
        /// <param name="nEstimators"></param>
        /// <param name="silent"></param>
        /// <param name="objective"></param>
        /// <param name="nThread"></param>
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
        public RegressionXGBoostLearner(int maxDepth = 3, float learningRate = 0.1F, int nEstimators = 100,
            bool silent = true, string objective = "reg:linear",
            int nThread = -1, float gamma = 0, int minChildWeight = 1,
            int maxDeltaStep = 0, float subsample = 1, float colSampleByTree = 1,
            float colSampleByLevel = 1, float regAlpha = 0, float regLambda = 1,
            float scalePosWeight = 1, float baseScore = 0.5F, int seed = 0,
            float missing = float.NaN)
        {
            m_parameters[ParameterNames.MaxDepth] = maxDepth;
            m_parameters[ParameterNames.LearningRate] = learningRate;
            m_parameters[ParameterNames.Estimators] = nEstimators;
            m_parameters[ParameterNames.Silent] = silent;
            m_parameters[ParameterNames.objective] = objective;

            m_parameters[ParameterNames.Threads] = nThread;
            m_parameters[ParameterNames.Gamma] = gamma;
            m_parameters[ParameterNames.MinChildWeight] = minChildWeight;
            m_parameters[ParameterNames.MaxDeltaStep] = maxDeltaStep;
            m_parameters[ParameterNames.SubSample] = subsample;
            m_parameters[ParameterNames.ColSampleByTree] = colSampleByTree;
            m_parameters[ParameterNames.ColSampleByLevel] = colSampleByLevel;
            m_parameters[ParameterNames.RegAlpha] = regAlpha;
            m_parameters[ParameterNames.RegLambda] = regLambda;
            m_parameters[ParameterNames.ScalePosWeight] = scalePosWeight;

            m_parameters[ParameterNames.BaseScore] = baseScore;
            m_parameters[ParameterNames.Seed] = seed;
            m_parameters[ParameterNames.Missing] = missing;
            m_parameters[ParameterNames.Booster] = null;
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
