using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingTestSplitters
{
    [TestClass]
    public class StratifiedTrainingTestIndexSplitterTest
    {
        [TestMethod]
        public void StratifiedTrainingTestIndexSplitter_Split()
        {
            var sut = new StratifiedTrainingTestIndexSplitter<double>(0.8);

            var targets = new double[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };

            var actual = sut.Split(targets);
            var expected = new TrainingTestIndexSplit(new int[] { 9, 0, 4, 2, 5, 7, 3, 8 },
                new int[] { 1, 6 });

            Assert.AreEqual(expected, actual);
        }
    }
}
