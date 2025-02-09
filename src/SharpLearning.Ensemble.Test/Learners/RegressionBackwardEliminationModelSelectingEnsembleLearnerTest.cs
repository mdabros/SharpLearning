using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Test.Learners;

[TestClass]
public class RegressionBackwardEliminationModelSelectingEnsembleLearnerTest
{
    [TestMethod]
    public void RegressionBackwardEliminationModelSelectingEnsembleLearner_Learn()
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

        var sut = new RegressionBackwardEliminationModelSelectingEnsembleLearner(learners, 5);

        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var model = sut.Learn(observations, targets);
        var predictions = model.Predict(observations);

        var metric = new MeanSquaredErrorRegressionMetric();
        var actual = metric.Error(targets, predictions);

        Assert.AreEqual(0.010316259438112841, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionBackwardEliminationModelSelectingEnsembleLearner_CreateMetaFeatures_Then_Learn()
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

        var sut = new RegressionBackwardEliminationModelSelectingEnsembleLearner(learners, 5);

        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var metaObservations = sut.LearnMetaFeatures(observations, targets);
        var model = sut.SelectModels(observations, metaObservations, targets);

        var predictions = model.Predict(observations);

        var metric = new MeanSquaredErrorRegressionMetric();
        var actual = metric.Error(targets, predictions);

        Assert.AreEqual(0.010316259438112841, actual, 0.0001);
    }

    [TestMethod]
    public void RegressionBackwardEliminationModelSelectingEnsembleLearner_Learn_Indexed()
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

        var sut = new RegressionBackwardEliminationModelSelectingEnsembleLearner(learners, 5);

        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var indices = Enumerable.Range(0, 25).ToArray();

        var model = sut.Learn(observations, targets, indices);
        var predictions = model.Predict(observations);

        var metric = new MeanSquaredErrorRegressionMetric();
        var actual = metric.Error(targets, predictions);

        Assert.AreEqual(0.13601421174394385, actual, 0.0001);
    }
}
