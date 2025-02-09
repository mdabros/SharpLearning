using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Metrics.Test.Regression;

[TestClass]
public class MeanAbsolutErrorRegressionMetricTest
{
    [TestMethod]
    public void MeanAbsolutErrorRegressionMetric_Error()
    {
        var targets = new double[] { 1.0, 2.3, 3.1, 4.4, 5.8 };
        var predictions = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var sut = new MeanAbsolutErrorRegressionMetric();

        var actual = sut.Error(targets, predictions);
        Assert.AreEqual(0.32, actual, 0.00001);
    }

    [TestMethod]
    public void MeanAbsolutErrorRegressionMetric_Error_Zero_Error()
    {
        var targets = new double[] { 0.0, 0.0 };
        var predictions = new double[] { 0.0, 0.0 };
        var sut = new MeanAbsolutErrorRegressionMetric();

        var actual = sut.Error(targets, predictions);
        Assert.AreEqual(0.0, actual);
    }
}
