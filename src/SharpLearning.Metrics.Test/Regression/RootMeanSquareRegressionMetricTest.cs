using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Metrics.Test.Regression;

[TestClass]
public class RootMeanSquareRegressionMetricTest
{
    [TestMethod]
    public void RootMeanSquareRegressionMetric_Error()
    {
        var targets = new double[] { 1.0, 2.3, 3.1, 4.4, 5.8 };
        var predictions = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var sut = new RootMeanSquareRegressionMetric();

        var result = sut.Error(targets, predictions);
        Assert.AreEqual(0.42426406871192851, result, 0.00001);
    }

    [TestMethod]
    public void RootMeanSquareRegressionMetric_Error_Zero_Error()
    {
        var targets = new double[] { 0.0, 0.0 };
        var predictions = new double[] { 0.0, 0.0 };
        var sut = new RootMeanSquareRegressionMetric();

        var result = sut.Error(targets, predictions);
        Assert.AreEqual(0.0, result);
    }
}
