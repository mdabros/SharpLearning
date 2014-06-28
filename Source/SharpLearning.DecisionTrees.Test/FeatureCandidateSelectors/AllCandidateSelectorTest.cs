using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.FeatureCandidateSelectors;
using SharpLearning.DecisionTrees.Learners;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Test.FeatureCandidateSelectors
{
    [TestClass]
    public class AllCandidateSelectorTest
    {
        [TestMethod]
        public void AllCandidateSelector_Select()
        {
            var featureIndices = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var sut = new AllFeatureCandidateSelector();
            var actual = new List<int>();
            sut.Select(featureIndices.Count, featureIndices.Count, actual);

            CollectionAssert.AreEqual(featureIndices, actual);
        }
    }
}
