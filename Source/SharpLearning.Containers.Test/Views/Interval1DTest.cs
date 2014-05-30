using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;
using System;

namespace SharpLearning.Containers.Test.Views
{
    [TestClass]
    public class Interval1DTest
    {
        [TestMethod]
        public void Interval1D_Equals()
        {
            var sut = Interval1D.Create(3, 5);
            var equal = Interval1D.Create(3, 5);
            var notEqual = Interval1D.Create(3, 4);

            Assert.IsTrue(sut.Equals(equal));
            Assert.IsFalse(sut.Equals(notEqual));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void Interval1D_InvalidArguments()
        {
            new Interval1D(5, 2);
        }
    }
}
