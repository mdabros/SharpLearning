using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.Metrics.Entropy;
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Test.SplitSearchers
{
    [TestClass]
    public class LinearSplitSearcherTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void LinearSplitSearcher_MinimumSplitSize()
        {
            new LinearSplitSearcher(-1);
        }

        [TestMethod]
        public void LinearSplitSearcher_FindBestSplit()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var feature = parser.EnumerateRows("AptitudeTestScore").ToF64Vector();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var entropyMetric = new GiniImpurityMetric();
            var initialResult = FindSplitResult.Initial();

            var sut = new LinearSplitSearcher(1);
            var parentEntropy = entropyMetric.Entropy(targets);

            var actual = sut.FindBestSplit(initialResult, 0, feature, targets, entropyMetric,
                Interval1D.Create(0, feature.Length), parentEntropy);

            var expected = new FindSplitResult(true, 4, 0.053792361484669093, new FeatureSplit(2.5, 0),
                new IntervalEntropy(Interval1D.Create(0, 4), 0), new IntervalEntropy(Interval1D.Create(4, 26), 0.49586776859504134));

            Assert.AreEqual(expected, actual);
        }
    }
}
