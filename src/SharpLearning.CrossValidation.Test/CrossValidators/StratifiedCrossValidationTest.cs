using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Test;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.CrossValidation.CrossValidators.Test;

/// <summary>
/// Summary description for StratifiedCrossValidationTest
/// </summary>
[TestClass]
public class StratifiedCrossValidationTest
{
    [TestMethod]
    public void StratisfiedCrossValidation_CrossValidate_Folds_2()
    {
        var actual = CrossValidate(2);
        Assert.AreEqual(0.34615384615384615, actual, 0.001);
    }

    [TestMethod]
    public void StratisfiedCrossValidation_CrossValidate_Folds_5()
    {
        var actual = CrossValidate(5);
        Assert.AreEqual(0.42307692307692307, actual, 0.001);
    }

    [TestMethod]
    public void StratisfiedCrossValidation_CrossValidate_Provide_Indices_Folds_2()
    {
        var actual = CrossValidate_Provide_Indices(2);
        Assert.AreEqual(0.69230769230769229, actual, 0.001);
    }

    [TestMethod]
    public void StratisfiedCrossValidation_CrossValidate_Provide_Indices_Folds_5()
    {
        var actual = CrossValidate_Provide_Indices(5);
        Assert.AreEqual(0.61538461538461542, actual, 0.001);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void StratisfiedCrossValidation_CrossValidate_Too_Many_Folds()
    {
        CrossValidate(200);
    }

    static double CrossValidate(int folds)
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var sut = new StratifiedCrossValidation<double>(folds, 42);
        var learner = new ClassificationDecisionTreeLearner();
        var predictions = sut.CrossValidate(learner, observations, targets);
        var metric = new TotalErrorClassificationMetric<double>();

        return metric.Error(targets, predictions);
    }

    static double CrossValidate_Provide_Indices(int folds)
    {
        var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

        var sut = new StratifiedCrossValidation<double>(folds, 42);

        var rowsToCrossvalidate = targets.Length / 2;
        var indices = Enumerable.Range(0, rowsToCrossvalidate).ToArray();
        var predictions = new double[rowsToCrossvalidate];
        var learner = new ClassificationDecisionTreeLearner();
        sut.CrossValidate(learner, observations, targets, indices, predictions);
        var metric = new TotalErrorClassificationMetric<double>();

        return metric.Error(targets.Take(rowsToCrossvalidate).ToArray(), predictions);
    }
}
