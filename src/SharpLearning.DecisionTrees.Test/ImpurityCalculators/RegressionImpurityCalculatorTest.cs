using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;

namespace SharpLearning.DecisionTrees.Test.ImpurityCalculators;

[TestClass]
public class RegressionImpurityCalculatorTest
{
    [TestMethod]
    public void RegressionImpurityCalculator_ImpurityImprovement()
    {
        var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };

        var parentInterval = Interval1D.Create(0, values.Length);

        var sut = new RegressionImpurityCalculator();
        sut.Init([], values, [], parentInterval);
        var impurity = sut.NodeImpurity();

        sut.UpdateIndex(50);
        var improvement1 = sut.ImpurityImprovement(impurity);
        Assert.AreEqual(75.0, improvement1, 0.000001);

        sut.UpdateIndex(96);
        var improvement2 = sut.ImpurityImprovement(impurity);
        Assert.AreEqual(69.473379629629648, improvement2, 0.000001);
    }

    [TestMethod]
    public void RegressionImpurityCalculator_ImpurityImprovement_Weighted()
    {
        var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };

        var weights = values.Select(t => Weight(t)).ToArray();
        var parentInterval = Interval1D.Create(0, values.Length);

        var sut = new RegressionImpurityCalculator();
        sut.Init([], values, weights, parentInterval);
        var impurity = sut.NodeImpurity();

        sut.UpdateIndex(50);
        var improvement1 = sut.ImpurityImprovement(impurity);
        Assert.AreEqual(167.04545454545456, improvement1, 0.000001);

        sut.UpdateIndex(96);
        var improvement2 = sut.ImpurityImprovement(impurity);
        Assert.AreEqual(162.78860028860029, improvement2, 0.000001);
    }

    [TestMethod]
    public void RegressionImpurityCalculator_ChildImpurities()
    {
        var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };

        var parentInterval = Interval1D.Create(0, values.Length);

        var sut = new RegressionImpurityCalculator();
        sut.Init([], values, [], parentInterval);
        var impurity = sut.NodeImpurity();

        sut.UpdateIndex(50);
        var actual = sut.ChildImpurities();
        var expected = new ChildImpurities(0.0, -2.25);

        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void RegressionImpurityCalculator_NodeImpurity()
    {
        var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };

        var parentInterval = Interval1D.Create(0, values.Length);

        var sut = new RegressionImpurityCalculator();
        sut.Init([], values, [], parentInterval);

        sut.UpdateIndex(50);
        var actual = sut.NodeImpurity();

        Assert.AreEqual(0.66666666666666674, actual, 0.000001);
    }

    [TestMethod]
    public void RegressionImpurityCalculator_LeafValue_Weighted()
    {
        var values = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
        var weights = values.Select(t => Weight(t)).ToArray();
        var parentInterval = Interval1D.Create(0, values.Length);

        var sut = new RegressionImpurityCalculator();
        sut.Init([], values, weights, parentInterval);

        var impurity = sut.NodeImpurity();

        sut.UpdateIndex(50);
        var actual = sut.LeafValue();

        Assert.AreEqual(1.75, actual, 0.000001);
    }

    static double Weight(double t)
    {
        return t == 2.0 ? 10.0 : 1.0;
    }
}
