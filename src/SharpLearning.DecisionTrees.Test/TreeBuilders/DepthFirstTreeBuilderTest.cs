using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.TreeBuilders;
using System;

namespace SharpLearning.DecisionTrees.Test.TreeBuilders
{
    [TestClass]
    public class DepthFirstTreeBuilderTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DepthFirstTreeBuilder_InvalidMaximumTreeSize()
        {
            new DepthFirstTreeBuilder(0, 1, 0.1, 42,
                new LinearSplitSearcher(1),
                new GiniClassificationImpurityCalculator());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DepthFirstTreeBuilder_InvalidFeaturesPrSplit()
        {
            new DepthFirstTreeBuilder(1, -1, 0.1, 42,
                new LinearSplitSearcher(1),
                new GiniClassificationImpurityCalculator());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DepthFirstTreeBuilder_InvalidMinimumInformationGain()
        {
            new DepthFirstTreeBuilder(1, 1, 0, 42,
                new LinearSplitSearcher(1),
                new GiniClassificationImpurityCalculator());
        }
    }
}
