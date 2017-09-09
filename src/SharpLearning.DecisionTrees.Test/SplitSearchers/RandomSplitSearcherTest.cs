using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using System;
using System.IO;
using System.Linq;

namespace SharpLearning.DecisionTrees.Test.SplitSearchers
{
    [TestClass]
    public class RandomSplitSearcherTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void OnlyUniqueThresholdsSplitSearcher_MinimumSplitSize()
        {
            new RandomSplitSearcher(-1, 42);
        }

        [TestMethod]
        public void RandomSplitSearcher_FindBestSplit()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var feature = parser.EnumerateRows("AptitudeTestScore").ToF64Vector();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var interval = Interval1D.Create(0, feature.Length);

            Array.Sort(feature, targets);

            var impurityCalculator = new GiniClassificationImpurityCalculator();
            impurityCalculator.Init(targets.Distinct().ToArray(), targets, new double[0], interval);
            var impurity = impurityCalculator.NodeImpurity();

            var sut = new RandomSplitSearcher(1, 42);

            var actual = sut.FindBestSplit(impurityCalculator, feature, targets,
                interval, impurity);

            var expected = new SplitResult(15, 3.6724258636461693, 0.037941545633853213,
                0.39111111111111119, 0.49586776859504134);

            Assert.AreEqual(expected, actual);
        }
    }
}
