using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Test.Learners
{
    [TestClass]
    public class RegressionRandomModelSelectingEnsembleLearnerTest
    {
        [TestMethod]
        public void RegressionRandomModelSelectingEnsembleLearner_Learn()
        {
            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9),
                new RegressionDecisionTreeLearner(11),
                new RegressionDecisionTreeLearner(21),
                new RegressionDecisionTreeLearner(23),
                new RegressionDecisionTreeLearner(1),
                new RegressionDecisionTreeLearner(14),
                new RegressionDecisionTreeLearner(17),
                new RegressionDecisionTreeLearner(19),
                new RegressionDecisionTreeLearner(33)

            };

            var sut = new RegressionRandomModelSelectingEnsembleLearner(learners, 5);

            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var metric = new MeanSquaredErrorRegressionMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.017238327841614508, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionRandomModelSelectingEnsembleLearner_Learn_Without_Replacement()
        {
            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9),
                new RegressionDecisionTreeLearner(11),
                new RegressionDecisionTreeLearner(21),
                new RegressionDecisionTreeLearner(23),
                new RegressionDecisionTreeLearner(1),
                new RegressionDecisionTreeLearner(14),
                new RegressionDecisionTreeLearner(17),
                new RegressionDecisionTreeLearner(19),
                new RegressionDecisionTreeLearner(33)

            };

            var metric = new MeanSquaredErrorRegressionMetric();

            var sut = new RegressionRandomModelSelectingEnsembleLearner(learners, 5,
                new RandomCrossValidation<double>(5, 42), new MeanRegressionEnsembleStrategy(), metric, 1, false);

            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.0090143589987671056, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionRandomModelSelectingEnsembleLearner_Learn_Start_With_3_Models()
        {
            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9),
                new RegressionDecisionTreeLearner(11),
                new RegressionDecisionTreeLearner(21),
                new RegressionDecisionTreeLearner(23),
                new RegressionDecisionTreeLearner(1),
                new RegressionDecisionTreeLearner(14),
                new RegressionDecisionTreeLearner(17),
                new RegressionDecisionTreeLearner(19),
                new RegressionDecisionTreeLearner(33)

            };

            var metric = new MeanSquaredErrorRegressionMetric();

            var sut = new RegressionRandomModelSelectingEnsembleLearner(learners, 5,
                new RandomCrossValidation<double>(5, 42), new MeanRegressionEnsembleStrategy(), metric, 3, false);

            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.0090143589987671056, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionRandomModelSelectingEnsembleLearner_Learn_Indexed()
        {
            var learners = new IIndexedLearner<double>[]
            {
                new RegressionDecisionTreeLearner(2),
                new RegressionDecisionTreeLearner(5),
                new RegressionDecisionTreeLearner(7),
                new RegressionDecisionTreeLearner(9),
                new RegressionDecisionTreeLearner(11),
                new RegressionDecisionTreeLearner(21),
                new RegressionDecisionTreeLearner(23),
                new RegressionDecisionTreeLearner(1),
                new RegressionDecisionTreeLearner(14),
                new RegressionDecisionTreeLearner(17),
                new RegressionDecisionTreeLearner(19),
                new RegressionDecisionTreeLearner(33)
            };

            var sut = new RegressionRandomModelSelectingEnsembleLearner(learners, 5);

            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var indices = Enumerable.Range(0, 25).ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);

            var metric = new MeanSquaredErrorRegressionMetric();
            var actual = metric.Error(targets, predictions);

            Assert.AreEqual(0.13601421174394385, actual, 0.0001);
        }
    }
}
