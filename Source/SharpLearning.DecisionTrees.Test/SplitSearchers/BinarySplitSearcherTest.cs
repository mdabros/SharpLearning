using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.Metrics.Entropy;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.Containers.Views;

namespace SharpLearning.DecisionTrees.Test.SplitSearchers
{
    [TestClass]
    public class BinarySplitSearcherTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BinarySplitSearcher_MinimumSplitSize()
        {
            new BinarySplitSearcher(-1);
        }

        [TestMethod]
        public void BinarySplitSearcher_FindBestSplit()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var feature = parser.EnumerateRows("AptitudeTestScore").ToF64Vector();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var entropyMetric = new GiniImpurityMetric();
            var initialResult = FindSplitResult.Initial();

            var sut = new BinarySplitSearcher(1);
            var parentEntropy = entropyMetric.Entropy(targets);

            var actual = sut.FindBestSplit(initialResult, 0, feature, targets, entropyMetric,
                Interval1D.Create(0, feature.Length), parentEntropy);

            var expected = new FindSplitResult(true, 9, 0.22222222222222221, new FeatureSplit(3.5, 0),
                new IntervalEntropy(Interval1D.Create(6, 8), 0), new IntervalEntropy(Interval1D.Create(9, 12), 0.44444444444444442));

            Assert.AreEqual(expected, actual);
        }
        
        [TestMethod]
        public void BinarySplitSearcher_FindBestSplit_DecisionTreeData()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var feature = parser.EnumerateRows("F2").ToF64Vector();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var entropyMetric = new NaiveSinglePassVarianceEntropyMetric();
            var initialResult = FindSplitResult.Initial();

            var sut = new BinarySplitSearcher(1);
            var parentEntropy = entropyMetric.Entropy(targets);

            var actual = sut.FindBestSplit(initialResult, 0, feature, targets, entropyMetric,
                Interval1D.Create(0, feature.Length), parentEntropy);

            var expected = new FindSplitResult(true, 36, 0.32383331576122876, new FeatureSplit(0.5676745, 0),
                new IntervalEntropy(Interval1D.Create(24, 35), 2.6550110844872536), new IntervalEntropy(Interval1D.Create(36, 48), 2.1261168581472947));

            Assert.AreEqual(expected, actual);
        }
    }
}
