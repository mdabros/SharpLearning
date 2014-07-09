using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.SplitSearchers;

namespace SharpLearning.DecisionTrees.Test.SplitSearchers
{
    [TestClass]
    public class FeatureSplitTest
    {
        [TestMethod]
        public void FeatureSplit_Equals()
        {
            var sut = new FeatureSplit(1.123213, 2);

            var equals = new FeatureSplit(1.123213, 2);
            var notEquals1 = new FeatureSplit(1.1, 2);
            var notEquals2 = new FeatureSplit(1.123213, 3);

            Assert.AreEqual(equals, sut);
            Assert.AreNotEqual(notEquals1, sut);
            Assert.AreNotEqual(notEquals2, sut);
        }
    }
}
