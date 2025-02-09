using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;

namespace SharpLearning.Containers.Test.Views;

[TestClass]
public class Interval2DTest
{
    [TestMethod]
    public void Interval2D_Equals()
    {
        var sut = Interval2D.Create(Interval1D.Create(0, 5), Interval1D.Create(2, 5));
        var equal = Interval2D.Create(Interval1D.Create(0, 5), Interval1D.Create(2, 5));
        var notEqual = Interval2D.Create(Interval1D.Create(2, 5), Interval1D.Create(0, 5));

        Assert.IsTrue(sut.Equals(equal));
        Assert.IsFalse(sut.Equals(notEqual));
    }
}
