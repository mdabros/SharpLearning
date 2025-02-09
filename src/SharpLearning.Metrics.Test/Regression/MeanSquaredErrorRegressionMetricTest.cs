using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Metrics.Test.Regression;

[TestClass]
public class MeanSquaredErrorRegressionMetricTest
{
    [TestMethod]
    public void MeanSquaredErrorRegressionMetric_Error_Zero_Error()
    {
        var targets = new double[] { 0, 0, 0, 0, 0, 0 };
        var predictions = new double[] { 0, 0, 0, 0, 0, 0 };

        var sut = new MeanSquaredErrorRegressionMetric();
        var actual = sut.Error(targets, predictions);

        Assert.AreEqual(0.0, actual);
    }

    [TestMethod]
    public void MeanSquaredErrorRegressionMetric_Error()
    {
        var targets = new double[] { 1.0, 2.3, 3.1, 4.4, 5.8 };
        var predictions = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        var sut = new MeanSquaredErrorRegressionMetric();
        var actual = sut.Error(targets, predictions);

        Assert.AreEqual(0.18, actual, 0.00001);
    }
}
