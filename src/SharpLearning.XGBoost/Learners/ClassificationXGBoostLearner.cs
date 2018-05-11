using System;
using System.Collections.Generic;
using System.Linq;
using SharpLearning.Common.Interfaces;
using SharpLearning.Containers.Matrices;
using SharpLearning.XGBoost.Models;
using XGBoost.lib;

namespace SharpLearning.XGBoost.Learners
{
    /// <summary>
    /// Classification learner for XGBoost
    /// </summary>
    public sealed class ClassificationXGBoostLearner : ILearner<double>
    {
        IDictionary<string, object> m_parameters = new Dictionary<string, object>();

        /// <summary>
        /// Classification learner for XGBoost.
        /// </summary>
        /// <param name="maximumTreeDepth">Maximum tree depth for base learners. (default is 3)</param>
        /// <param name="learningRate">Boosting learning rate (xgb's "eta"). 0 indicates no limit. (default is 0.1)</param>
        /// <param name="estimators">Number of estimators to fit. (default is 100)</param>
        /// <param name="silent">Whether to print messages while running boosting. (default is false)</param>
        /// <param name="objective">Specify the learning task and the corresponding learning objective. (default is softmax)</param>
        /// <param name="boosterType"> which booster to use, can be gbtree, gblinear or dart. 
        /// gbtree and dart use tree based model while gblinear uses linear function (default is gbtree)</param>
        /// <param name="treeMethod">The tree construction algorithm used in XGBoost. See reference paper: https://arxiv.org/abs/1603.02754. (default is auto)</param>
        /// <param name="numberOfThreads">Number of parallel threads used to run xgboost. -1 means use all thread avialable. (default is -1)</param>
        /// <param name="gamma">Minimum loss reduction required to make a further partition on a leaf node of the tree. (default is 0) </param>
        /// <param name="minChildWeight">Minimum sum of instance weight(hessian) needed in a child. (default is 1)</param>
        /// <param name="maxDeltaStep">Maximum delta step we allow each tree's weight estimation to be. (default is 0)</param>
        /// <param name="subsample">Subsample ratio of the training instance. (default is 1)</param>
        /// <param name="colSampleByTree">Subsample ratio of columns when constructing each tree. (defualt is 1)</param>
        /// <param name="colSampleByLevel">Subsample ratio of columns for each split, in each level. (defualt is 1)</param>
        /// <param name="l1Regularization">L1 regularization term on weights. Also known as RegAlpha. (default is 0)</param>
        /// <param name="l2Reguralization">L2 regularization term on weights. Also known as regLambda. (default is 1)</param>
        /// <param name="scalePosWeight">Balancing of positive and negative weights. (default is 1)</param>
        /// <param name="baseScore">The initial prediction score of all instances, global bias. (default is 0.5)</param>
        /// <param name="seed">Random number seed. (defaukt is 0)</param>
        /// <param name="missing">Value in the data which needs to be present as a missing value. (default is NaN)</param>
        public ClassificationXGBoostLearner(int maximumTreeDepth = 3, double learningRate = 0.1, int estimators = 100,
            bool silent = true,
            Objective objective = Objective.Softmax,
            BoosterType boosterType = BoosterType.GBTree,
            TreeMethod treeMethod = TreeMethod.Auto,
            int numberOfThreads = -1, double gamma = 0, int minChildWeight = 1,
            int maxDeltaStep = 0, double subsample = 1, double colSampleByTree = 1,
            double colSampleByLevel = 1, double l1Regularization = 0, double l2Reguralization = 1,
            double scalePosWeight = 1, double baseScore = 0.5, int seed = 0,
            double missing = double.NaN)
        {
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(maximumTreeDepth), maximumTreeDepth, 0);
            ArgumentChecks.ThrowOnArgumentLessThanOrHigherThan(nameof(learningRate), learningRate, 0, 1.0);
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(estimators), estimators, 1);
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(numberOfThreads), numberOfThreads, -1);
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(gamma), gamma, 0);
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(minChildWeight), minChildWeight, 0);
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(maxDeltaStep), maxDeltaStep, 0);
            ArgumentChecks.ThrowOnArgumentLessThanOrHigherThan(nameof(subsample), subsample, 0, 1.0);
            ArgumentChecks.ThrowOnArgumentLessThanOrHigherThan(nameof(colSampleByTree), colSampleByTree, 0, 1.0);
            ArgumentChecks.ThrowOnArgumentLessThanOrHigherThan(nameof(colSampleByLevel), colSampleByLevel, 0, 1.0);
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(l1Regularization), l1Regularization, 0);
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(l2Reguralization), l2Reguralization, 0);
            ArgumentChecks.ThrowOnArgumentLessThan(nameof(scalePosWeight), scalePosWeight, 0);

            m_parameters[ParameterNames.MaxDepth] = maximumTreeDepth;
            m_parameters[ParameterNames.LearningRate] = (float)learningRate;
            m_parameters[ParameterNames.Estimators] = estimators;
            m_parameters[ParameterNames.Silent] = silent;
            m_parameters[ParameterNames.objective] = objective.ToXGBoostString();

            m_parameters[ParameterNames.Threads] = numberOfThreads;
            m_parameters[ParameterNames.Gamma] = (float)gamma;
            m_parameters[ParameterNames.MinChildWeight] = minChildWeight;
            m_parameters[ParameterNames.MaxDeltaStep] = maxDeltaStep;
            m_parameters[ParameterNames.SubSample] = (float)subsample;
            m_parameters[ParameterNames.ColSampleByTree] = (float)colSampleByTree;
            m_parameters[ParameterNames.ColSampleByLevel] = (float)colSampleByLevel;
            m_parameters[ParameterNames.RegAlpha] = (float)l1Regularization;
            m_parameters[ParameterNames.RegLambda] = (float)l2Reguralization;
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
        public ClassificationXGBoostModel Learn(F64Matrix observations, double[] targets)
        {
            var floatObservations = observations.ToFloatJaggedArray();

            var index = 0;
            var targetNameToIndex = targets.Distinct()
                .Select(v => v)
                .OrderBy(v => v)
                .ToDictionary(v => v, v => (float)index++);

            var numberOfClasses = targetNameToIndex.Count();
            m_parameters[ParameterNames.NumberOfClasses] = numberOfClasses;

            var floatTargets = targets.Select(v => targetNameToIndex[v]).ToArray();

            using (var train = new DMatrix(floatObservations, floatTargets))
            {
                var booster = new Booster(m_parameters.ToDictionary(v => v.Key, v => v.Value), train);
                var iterations = (int)m_parameters[ParameterNames.Estimators];

                for (int iteration = 0; iteration < iterations; iteration++)
                {
                    booster.Update(train, iteration);
                }

                return new ClassificationXGBoostModel(booster, targetNameToIndex.ToDictionary(v => v.Value, v => v.Key));
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
