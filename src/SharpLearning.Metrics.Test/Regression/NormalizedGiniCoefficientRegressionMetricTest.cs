using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Metrics.Test.Regression;

[TestClass]
public class NormalizedGiniCoefficientRegressionMetricTest
{
    [TestMethod]
    public void MeanAbsolutErrorRegressionMetric_Error()
    {
        var targets = new double[] { 1.0, 2.3, 3.1, 4.4, 5.8 };
        var predictions = new double[] { 1.0, 2.0, 3.0, 1000.0, 5.0 };
        var sut = new NormalizedGiniCoefficientRegressionMetric();

        var actual = sut.Error(targets, predictions);
        Assert.AreEqual(0.11965811965811968, actual, 0.00001);
    }

    [TestMethod]
    public void MeanAbsolutErrorRegressionMetric_Zero_Error()
    {
        var targets = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var predictions = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var sut = new NormalizedGiniCoefficientRegressionMetric();

        var actual = sut.Error(targets, predictions);
        Assert.AreEqual(0.0, actual, 0.00001);
    }
}
