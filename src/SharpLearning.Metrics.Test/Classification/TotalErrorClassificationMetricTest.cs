using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Metrics.Test.Classification;

[TestClass]
public class TotalErrorClassificationMetricTest
{
    [TestMethod]
    public void TotalErrorClassificationMetric_Error()
    {
        var predictions = new double[] { 0, 1, 1, 2, 3, 4, 4 };
        var targets = new double[] { 0, 1, 1, 2, 2, 3, 4 };

        var sut = new TotalErrorClassificationMetric<double>();
        var actual = sut.Error(targets, predictions);

        Assert.AreEqual(0.2857142857142857, actual);
    }

    [TestMethod]
    public void TotalErrorClassificationMetric_Error_Zero_Error()
    {
        var predictions = new double[] { 0, 1, 1, 2, 3, 4, 4 };
        var targets = new double[] { 0, 1, 1, 2, 3, 4, 4 };

        var sut = new TotalErrorClassificationMetric<double>();
        var actual = sut.Error(targets, predictions);

        Assert.AreEqual(0.0, actual);
    }

    [TestMethod]
    public void TotalErrorClassificationMetric_Error_All_Error()
    {
        var predictions = new double[] { 1, 1, 1, 1 };
        var targets = new double[] { 0, 0, 0, 0 };

        var sut = new TotalErrorClassificationMetric<double>();
        var actual = sut.Error(targets, predictions);

        Assert.AreEqual(1.0, actual);
    }

    [TestMethod]
    public void TotalErrorClassificationMetric_ErrorString()
    {
        var predictions = new double[] { 0, 1, 1, 2, 3, 4, 4 };
        var targets = new double[] { 0, 1, 1, 2, 2, 3, 4 };

        var sut = new TotalErrorClassificationMetric<double>();
        var actual = sut.ErrorString(targets, predictions);

        var expected = ";0;1;2;3;4;0;1;2;3;4\r\n0;1.000;0.000;0.000;0.000;0.000;100.000;0.000;0.000;0.000;0.000\r\n1;0.000;2.000;0.000;0.000;0.000;0.000;100.000;0.000;0.000;0.000\r\n2;0.000;0.000;1.000;1.000;0.000;0.000;0.000;50.000;50.000;0.000\r\n3;0.000;0.000;0.000;0.000;1.000;0.000;0.000;0.000;0.000;100.000\r\n4;0.000;0.000;0.000;0.000;1.000;0.000;0.000;0.000;0.000;100.000\r\nError: 28.571\r\n";
        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void TotalErrorClassificationMetric_ErrorString_TargetStringMapping()
    {
        var predictions = new double[] { 0, 1, 1, 2, 3, 4, 4 };
        var targets = new double[] { 0, 1, 1, 2, 2, 3, 4 };

        var sut = new TotalErrorClassificationMetric<double>();
        var targetStringMapping = new Dictionary<double, string> { { 0, "One" }, { 1, "Two" }, { 2, "Three" }, { 3, "Four" }, { 4, "Five" } };

        var actual = sut.ErrorString(targets, predictions, targetStringMapping);
        var expected = ";One;Two;Three;Four;Five;One;Two;Three;Four;Five\r\nOne;1.000;0.000;0.000;0.000;0.000;100.000;0.000;0.000;0.000;0.000\r\nTwo;0.000;2.000;0.000;0.000;0.000;0.000;100.000;0.000;0.000;0.000\r\nThree;0.000;0.000;1.000;1.000;0.000;0.000;0.000;50.000;50.000;0.000\r\nFour;0.000;0.000;0.000;0.000;1.000;0.000;0.000;0.000;0.000;100.000\r\nFive;0.000;0.000;0.000;0.000;1.000;0.000;0.000;0.000;0.000;100.000\r\nError: 28.571\r\n";

        Assert.AreEqual(expected, actual);
    }
}
