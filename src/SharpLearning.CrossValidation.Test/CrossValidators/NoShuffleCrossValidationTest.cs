using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Test;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.CrossValidation.CrossValidators.Test;

[TestClass]
public class NoShuffleCrossValidationTest
{
    [TestMethod]
    public void NoShuffleCrossValidation_CrossValidate_Folds_2()
    {
        var actual = CrossValidate(2);
        Assert.AreEqual(0.08399278971163, actual, 0.001);
    }

    [TestMethod]
    public void NoShuffleCrossValidation_CrossValidate_Folds_10()
    {
        var actual = CrossValidate(10);
        Assert.AreEqual(0.069356782107075, actual, 0.001);
    }

    [TestMethod]
    public void NoShuffleCrossValidation_CrossValidate_Provide_Indices_Folds_2()
    {
        var actual = CrossValidate_Provide_Indices(2);
        Assert.AreEqual(0.10219923025847003, actual, 0.001);
    }

    [TestMethod]
    public void NoShuffleCrossValidation_CrossValidate_Provide_Indices_Folds_10()
    {
        var actual = CrossValidate_Provide_Indices(10);
        Assert.AreEqual(0.10969696400057005, actual, 0.001);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void NoShuffleCrossValidation_CrossValidate_Too_Many_Folds()
    {
        CrossValidate(2000);
    }

    static double CrossValidate(int folds)
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new NoShuffleCrossValidation<double>(folds);
        var learner = new RegressionDecisionTreeLearner();
        var predictions = sut.CrossValidate(learner, observations, targets);
        var metric = new MeanSquaredErrorRegressionMetric();

        return metric.Error(targets, predictions);
    }

    static double CrossValidate_Provide_Indices(int folds)
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new NoShuffleCrossValidation<double>(folds);

        var rowsToCrossvalidate = targets.Length / 2;
        var indices = Enumerable.Range(0, rowsToCrossvalidate).ToArray();
        var predictions = new double[rowsToCrossvalidate];

        var learner = new RegressionDecisionTreeLearner();
        sut.CrossValidate(learner, observations, targets, indices, predictions);
        var metric = new MeanSquaredErrorRegressionMetric();

        return metric.Error(targets.Take(rowsToCrossvalidate).ToArray(), predictions);
    }
}
