using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingTestSplitters
{
    [TestClass]
    public class TrainingTestIndexSplitTest
    {
        [TestMethod]
        public void TrainingTestIndexSplit_Equals()
        {
            var sut = new TrainingTestIndexSplit(new int[] { 1, 2 }, new int[] { 3, 4 });

            var equal = new TrainingTestIndexSplit(new int[] { 1, 2 }, new int[] { 3, 4 });
            var notEqual = new TrainingTestIndexSplit(new int[] { 3, 4 }, new int[] { 1, 2 });

            Assert.IsTrue(sut.Equals(equal));
            Assert.IsFalse(sut.Equals(notEqual));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TrainingTestIndexSplit_TrainingIndices_Is_Null()
        {
            new TrainingTestIndexSplit(null, new int[] { 3, 4 });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TrainingTestIndexSplit_ValidationIndices_Is_Null()
        {
            new TrainingTestIndexSplit(new int[] { 3, 4 }, null);
        }
    }
}
