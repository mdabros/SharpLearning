using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingValidationSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingValidationSplitters
{
    [TestClass]
    public class NoShuffleTrainingValidationIndexSplitterTest
    {
        [TestMethod]
        public void NoShuffleTrainingValidationIndexSplitter_Split()
        {
            var sut = new NoShuffleTrainingValidationIndexSplitter<double>(0.8);

            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            var actual = sut.Split(targets);
            var expected = new TrainingValidationIndexSplit(new int[] { 0, 1, 2, 3, 4, 5, 6, 7 },
                new int[] { 8, 9 });

            Assert.AreEqual(expected, actual);
        }
    }
}
