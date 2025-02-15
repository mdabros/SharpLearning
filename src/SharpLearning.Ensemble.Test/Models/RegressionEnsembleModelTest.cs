using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Common.Interfaces;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.Ensemble.Strategies;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Ensemble.Test.Models;

[TestClass]
public class RegressionEnsembleModelTest
{
    [TestMethod]
    public void RegressionEnsembleModel_Predict_single()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learners = new IIndexedLearner<double>[]
        {
            new RegressionDecisionTreeLearner(2),
            new RegressionDecisionTreeLearner(5),
            new RegressionDecisionTreeLearner(7),
            new RegressionDecisionTreeLearner(9),
        };

        var learner = new RegressionEnsembleLearner(learners, new MeanRegressionEnsembleStrategy());
        var sut = learner.Learn(observations, targets);

        var rows = targets.Length;
        var predictions = new double[rows];
        for (var i = 0; i < rows; i++)
        {
            predictions[i] = sut.Predict(observations.Row(i));
        }

        var metric = new MeanSquaredErrorRegressionMetric();
        var actual = metric.Error(targets, predictions);

        Assert.AreEqual(0.033195970695970689, actual, 0.0000001);
    }

    [TestMethod]
    public void RegressionEnsembleModel_Predict_Multiple()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learners = new IIndexedLearner<double>[]
        {
            new RegressionDecisionTreeLearner(2),
            new RegressionDecisionTreeLearner(5),
            new RegressionDecisionTreeLearner(7),
            new RegressionDecisionTreeLearner(9),
        };

        var learner = new RegressionEnsembleLearner(learners, new MeanRegressionEnsembleStrategy());
        var sut = learner.Learn(observations, targets);

        var predictions = sut.Predict(observations);

        var metric = new MeanSquaredErrorRegressionMetric();
        var actual = metric.Error(targets, predictions);

        Assert.AreEqual(0.033195970695970689, actual, 0.0000001);
    }

    [TestMethod]
    public void RegressionEnsembleModel_GetVariableImportance()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 },
            { "PreviousExperience_month", 1 }, };

        var learners = new IIndexedLearner<double>[]
        {
            new RegressionDecisionTreeLearner(2),
            new RegressionDecisionTreeLearner(5),
            new RegressionDecisionTreeLearner(7),
            new RegressionDecisionTreeLearner(9),
        };

        var learner = new RegressionEnsembleLearner(learners, new MeanRegressionEnsembleStrategy());
        var sut = learner.Learn(observations, targets);

        var actual = sut.GetVariableImportance(featureNameToIndex);
        var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 },
            { "AptitudeTestScore", 3.46067371526717 }, };

        Assert.AreEqual(expected.Count, actual.Count);
        var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

        foreach (var item in zip)
        {
            Assert.AreEqual(item.Expected.Key, item.Actual.Key);
            Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
        }
    }

    [TestMethod]
    public void RegressionEnsembleModel_GetRawVariableImportance()
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var learners = new IIndexedLearner<double>[]
        {
            new RegressionDecisionTreeLearner(2),
            new RegressionDecisionTreeLearner(5),
            new RegressionDecisionTreeLearner(7),
            new RegressionDecisionTreeLearner(9),
        };

        var learner = new RegressionEnsembleLearner(learners, new MeanRegressionEnsembleStrategy());
        var sut = learner.Learn(observations, targets);

        var actual = sut.GetRawVariableImportance();
        var expected = new double[] { 100.0, 3.46067371526717 };

        Assert.AreEqual(expected.Length, actual.Length);

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], 0.000001);
        }
    }
}
