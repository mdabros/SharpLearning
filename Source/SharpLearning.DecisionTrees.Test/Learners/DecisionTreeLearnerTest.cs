using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Metrics.Entropy;
using SharpLearning.DecisionTrees.FeatureCandidateSelectors;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.ImpurityCalculators;

namespace SharpLearning.DecisionTrees.Test.Learners
{
    [TestClass]
    public class DecisionTreeLearnerTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DecisionTreeLearner_InvalidMaximumTreeSize()
        {
            new DecisionTreeLearner(0, 1, 0.1, 42,
                new LinearSplitSearcher(1),
                new GiniClasificationImpurityCalculator());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DecisionTreeLearner_InvalidFeaturesPrSplit()
        {
            new DecisionTreeLearner(1, 0, 0.1, 42,
                new LinearSplitSearcher(1),
                new GiniClasificationImpurityCalculator());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DecisionTreeLearner_InvalidMinimumInformationGain()
        {
            new DecisionTreeLearner(1, 1, 0, 42,
                new LinearSplitSearcher(1),
                new GiniClasificationImpurityCalculator());
        }
    }
}
