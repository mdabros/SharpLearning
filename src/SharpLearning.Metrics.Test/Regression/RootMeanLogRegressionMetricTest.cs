using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Metrics.Test.Regression;

[TestClass]
public class RootMeanLogRegressionMetricTest
{
    [TestMethod]
    public void RootMeanLogRegressionMetric_Error()
    {
        var targets = new double[] { 1.0, 0.5, 1.5, 2.0, 3.0, 2.0, 5.0, 1.0, 0.0, 10.0, 100.0, 2000.0, 5000.0 };
        var predictions = new double[] { 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 3.0, 5.0, 5.0, 5.0, 90.0, 1000.0, 10000.0 };
        var sut = new RootMeanLogRegressionMetric();

        var actual = sut.Error(targets, predictions);
        Assert.AreEqual(0.787833389, actual, 0.00001);
    }

    [TestMethod]
    public void RootMeanLogRegressionMetric_Error_Zero_Error()
    {
        var targets = new double[] { 0.0, 0.0 };
        var predictions = new double[] { 0.0, 0.0 };
        var sut = new RootMeanLogRegressionMetric();

        var actual = sut.Error(targets, predictions);
        Assert.AreEqual(0.0, actual);
    }
}
