using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Test.SplitSearchers
{
    [TestClass]
    public class IntervalEntropyTest
    {
        [TestMethod]
        public void IntervalEntropy_Equals()
        {
            var sut = new IntervalEntropy(Interval1D.Create(23, 55), 123.32542);

            var equal = new IntervalEntropy(Interval1D.Create(23, 55), 123.32542);
            var notEqual1 = new IntervalEntropy(Interval1D.Create(23, 234), 123.32542);
            var notEqual2 = new IntervalEntropy(Interval1D.Create(23, 55), 123.3);

            Assert.AreEqual(equal, sut);
            Assert.AreNotEqual(notEqual1, sut);
            Assert.AreNotEqual(notEqual2, sut);
        }
    }
}
