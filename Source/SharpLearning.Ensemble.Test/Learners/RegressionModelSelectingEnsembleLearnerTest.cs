using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.Ensemble.EnsembleSelectors;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Regression;
using System;

namespace SharpLearning.Ensemble.Test.Learners
{
    [TestClass]
    public class RegressionModelSelectingEnsembleLearnerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RegressionModelSelectingEnsembleLearner_Constructor_Learners_Null()
        {
            var metric = new MeanSquaredErrorRegressionMetric();
            var ensembleStrategy = new MeanRegressionEnsembleStrategy();
            var ensembleSelection = new ForwardSearchRegressionEnsembleSelection(metric, ensembleStrategy, 5, 1, true);
            var crossValidation = new RandomCrossValidation<double>(5);

            var sut = new RegressionModelSelectingEnsembleLearner(null, crossValidation, ensembleStrategy, ensembleSelection);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RegressionModelSelectingEnsembleLearner_Constructor_CrossValidation_Null()
        {
            var learners = new IIndexedLearner <double>[4];
            var metric = new MeanSquaredErrorRegressionMetric();
            var ensembleStrategy = new MeanRegressionEnsembleStrategy();
            var ensembleSelection = new ForwardSearchRegressionEnsembleSelection(metric, ensembleStrategy, 5, 1, true);

            var sut = new RegressionModelSelectingEnsembleLearner(learners, null, ensembleStrategy, ensembleSelection);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void RegressionModelSelectingEnsembleLearner_Constructor_EnsembleSelection_Null()
        {
            var learners = new IIndexedLearner<double>[4];
            var metric = new MeanSquaredErrorRegressionMetric();
            var ensembleStrategy = new MeanRegressionEnsembleStrategy();
            var crossValidation = new RandomCrossValidation<double>(5);

            var sut = new RegressionModelSelectingEnsembleLearner(learners, crossValidation, ensembleStrategy, null);
        }
    }
}
