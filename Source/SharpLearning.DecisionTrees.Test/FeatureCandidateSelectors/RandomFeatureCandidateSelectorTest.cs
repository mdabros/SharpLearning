using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.FeatureCandidateSelectors;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Test.FeatureCandidateSelectors
{
    [TestClass]
    public class RandomFeatureCandidateSelectorTest
    {
        [TestMethod]
        public void RandomFeatureCandidateSelector_Select_4()
        {
            var featureIndices = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var sut = new RandomFeatureCandidateSelector(42);
            var actual = new List<int>();
            sut.Select(4, featureIndices.Count, actual);

            var expected = new List<int> { 9, 0, 4, 2 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RandomFeatureCandidateSelector_Select_6()
        {
            var featureIndices = new List<int> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var sut = new RandomFeatureCandidateSelector(42);
            var actual = new List<int>();
            sut.Select(6, featureIndices.Count, actual);

            var expected = new List<int> { 9, 0, 4, 2, 5, 7 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void RandomFeatureCandidateSelector_Select_SelectCount_Check()
        {
            var sut = new RandomFeatureCandidateSelector(42);
            sut.Select(20, 10, new List<int>());
        }
    }
}
