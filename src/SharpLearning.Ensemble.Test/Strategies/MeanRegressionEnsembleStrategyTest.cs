using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Ensemble.Strategies;

namespace SharpLearning.Ensemble.Test.Strategies;

[TestClass]
public class MeanRegressionEnsembleStrategyTest
{
    [TestMethod]
    public void MeanRegressionEnsembleStrategy_Combine()
    {
        var sut = new MeanRegressionEnsembleStrategy();
        var values = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };
        var actual = sut.Combine(values);

        Assert.AreEqual(3.0, actual, 0.001);
    }
}
