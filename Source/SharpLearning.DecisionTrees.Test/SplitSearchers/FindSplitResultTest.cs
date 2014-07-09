using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Test.SplitSearchers
{
    [TestClass]
    public class FindSplitResultTest
    {
        [TestMethod]
        public void FindSplitResult_Equals()
        {
            var sut = new FindSplitResult(true, 1, 13.4543, new FeatureSplit(200, 4),
                new IntervalEntropy(Interval1D.Create(23, 54), 12.5), new IntervalEntropy(Interval1D.Create(23, 54), 10.5));

            var equal = new FindSplitResult(true, 1, 13.4543, new FeatureSplit(200, 4),
                new IntervalEntropy(Interval1D.Create(23, 54), 12.5), new IntervalEntropy(Interval1D.Create(23, 54), 10.5));

            var notEqual1 = new FindSplitResult(false, 1, 13.4543, new FeatureSplit(200, 4),
                new IntervalEntropy(Interval1D.Create(23, 54), 12.5), new IntervalEntropy(Interval1D.Create(23, 54), 10.5));

            var notEqual2 = new FindSplitResult(true, 2, 13.4543, new FeatureSplit(200, 4),
                new IntervalEntropy(Interval1D.Create(23, 54), 12.5), new IntervalEntropy(Interval1D.Create(23, 54), 10.5));

            var notEqual3 = new FindSplitResult(true, 1, 14.4543, new FeatureSplit(200, 4),
                new IntervalEntropy(Interval1D.Create(23, 54), 12.5), new IntervalEntropy(Interval1D.Create(23, 54), 10.5));

            var notEqual4 = new FindSplitResult(true, 1, 13.4543, new FeatureSplit(20, 4),
                new IntervalEntropy(Interval1D.Create(23, 54), 12.5), new IntervalEntropy(Interval1D.Create(23, 54), 10.5));

            var notEqual5 = new FindSplitResult(true, 1, 13.4543, new FeatureSplit(200, 4),
                new IntervalEntropy(Interval1D.Create(23, 540), 12.5), new IntervalEntropy(Interval1D.Create(23, 54), 10.5));

            var notEqual6 = new FindSplitResult(true, 1, 13.4543, new FeatureSplit(200, 4),
                new IntervalEntropy(Interval1D.Create(23, 54), 12.5), new IntervalEntropy(Interval1D.Create(23, 54), 100.5));

            Assert.AreEqual(equal, sut);
            Assert.AreNotEqual(notEqual1, sut);
            Assert.AreNotEqual(notEqual2, sut);
            Assert.AreNotEqual(notEqual3, sut);
            Assert.AreNotEqual(notEqual4, sut);
            Assert.AreNotEqual(notEqual5, sut);
            Assert.AreNotEqual(notEqual5, sut);
        }
    }
}
