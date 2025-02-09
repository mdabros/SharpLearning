using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Metrics.Test.ModelComparison;

[TestClass]
public class McNemarModelComparisonTest
{
    [TestMethod]
    public void McNemarModelComparison_Compare()
    {
        var targets = new double[] { 1, 2, 3, 4, 4, 4, 3, 3, 1, 1 };
        var model1 = new double[] { 2, 2, 3, 4, 3, 2, 3, 2, 1, 1 };
        var model2 = new double[] { 1, 1, 3, 4, 3, 4, 2, 3, 1, 1 };

        var actual = McNemarModelComparison.Compare(model1, model2, targets);

        CollectionAssert.AreEqual(new int[] { 1, 2 }, actual[0]);
        CollectionAssert.AreEqual(new int[] { 3, 4 }, actual[1]);
    }

    [TestMethod]
    public void McNemarModelComparison_CompareString()
    {
        var targets = new double[] { 1, 2, 3, 4, 4, 4, 3, 3, 1, 1 };
        var model1 = new double[] { 2, 2, 3, 4, 3, 2, 3, 2, 1, 1 };
        var model2 = new double[] { 1, 1, 3, 4, 3, 4, 2, 3, 1, 1 };

        var actual = McNemarModelComparison.CompareString(model1, model2, targets);
        var expected = ";Model1Wrong;Model1Right\r\nModel2Wrong;1;2\r\nModel2Right;3;4";

        Assert.AreEqual(expected, actual);
    }
}
