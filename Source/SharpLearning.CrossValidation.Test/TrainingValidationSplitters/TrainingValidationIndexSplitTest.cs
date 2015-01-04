using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingValidationSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingValidationSplitters
{
    [TestClass]
    public class TrainingValidationIndexSplitTest
    {
        [TestMethod]
        public void TrainingValidationIndexSplit_Equals()
        {
            var sut = new TrainingValidationIndexSplit(new int[] { 1, 2 }, new int[] { 3, 4 });

            var equal = new TrainingValidationIndexSplit(new int[] { 1, 2 }, new int[] { 3, 4 });
            var notEqual = new TrainingValidationIndexSplit(new int[] { 3, 4 }, new int[] { 1, 2 });

            Assert.IsTrue(sut.Equals(equal));
            Assert.IsFalse(sut.Equals(notEqual));
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TrainingValidationIndexSplit_TrainingIndices_Is_Null()
        {
            new TrainingValidationIndexSplit(null, new int[] { 3, 4 });
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TrainingValidationIndexSplit_ValidationIndices_Is_Null()
        {
            new TrainingValidationIndexSplit(new int[] { 3, 4 }, null);
        }
    }
}
