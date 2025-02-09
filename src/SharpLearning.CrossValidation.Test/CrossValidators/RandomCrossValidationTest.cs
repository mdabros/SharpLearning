using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Test;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.CrossValidation.CrossValidators.Test;

[TestClass]
public class RandomCrossValidationTest
{
    [TestMethod]
    public void RandomCrossValidation_CrossValidate_Folds_2()
    {
        var actual = CrossValidate(2);
        Assert.AreEqual(0.090240740378955, actual, 0.001);
    }

    [TestMethod]
    public void RandomCrossValidation_CrossValidate_Folds_10()
    {
        var actual = CrossValidate(10);
        Assert.AreEqual(0.068002901204319982, actual, 0.001);
    }

    [TestMethod]
    public void RandomCrossValidation_CrossValidate_Provide_Indices_Folds_2()
    {
        var actual = CrossValidate_Provide_Indices(2);
        Assert.AreEqual(0.09387129961716, actual, 0.001);
    }

    [TestMethod]
    public void RandomCrossValidation_CrossValidate_Provide_Indices_Folds_10()
    {
        var actual = CrossValidate_Provide_Indices(10);
        Assert.AreEqual(0.088594159444240025, actual, 0.001);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void RandomCrossValidation_CrossValidate_Too_Many_Folds()
    {
        CrossValidate(2000);
    }

    static double CrossValidate(int folds)
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RandomCrossValidation<double>(folds, 42);
        var learner = new RegressionDecisionTreeLearner();
        var predictions = sut.CrossValidate(learner, observations, targets);
        var metric = new MeanSquaredErrorRegressionMetric();

        return metric.Error(targets, predictions);
    }

    static double CrossValidate_Provide_Indices(int folds)
    {
        var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

        var sut = new RandomCrossValidation<double>(folds, 42);

        var rowsToCrossvalidate = targets.Length / 2;
        var indices = Enumerable.Range(0, rowsToCrossvalidate).ToArray();
        var predictions = new double[rowsToCrossvalidate];
        var learner = new RegressionDecisionTreeLearner();
        sut.CrossValidate(learner, observations, targets, indices, predictions);
        var metric = new MeanSquaredErrorRegressionMetric();

        return metric.Error(targets.Take(rowsToCrossvalidate).ToArray(), predictions);
    }
}
