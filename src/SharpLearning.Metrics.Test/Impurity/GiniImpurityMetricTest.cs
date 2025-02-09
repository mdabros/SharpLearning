using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Impurity;

namespace SharpLearning.Metrics.Test.Impurity;

[TestClass]
public class GiniImpurityMetricTest
{
    [TestMethod]
    public void GiniImpurityMetric_Impurity()
    {
        var set1 = new double[] { 0, 1, 2, 3, 4, 3, 2, 1, 0 };
        var set2 = new double[] { 1, 1, 1, 1, 2, 2, 2, 2 };
        var set3 = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };

        var sut = new GiniImpurityMetric();

        var val1 = sut.Impurity(set1);
        Assert.AreEqual(0.79012345679012341, val1);
        var val2 = sut.Impurity(set2);
        Assert.AreEqual(0.5, val2);
        var val3 = sut.Impurity(set3);
        Assert.AreEqual(0.0, val3);
    }
}
