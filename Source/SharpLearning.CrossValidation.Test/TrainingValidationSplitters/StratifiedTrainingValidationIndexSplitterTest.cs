using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingValidationSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingValidationSplitters
{
    [TestClass]
    public class StratifiedTrainingValidationIndexSplitterTest
    {
        [TestMethod]
        public void StratifiedTrainingValidationIndexSplitter_Split()
        {
            var sut = new StratifiedTrainingValidationIndexSplitter<double>(0.8);

            var targets = new double[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };

            var actual = sut.Split(targets);
            var expected = new TrainingValidationIndexSplit(new int[] { 9, 0, 4, 2, 5, 7, 3, 8 },
                new int[] { 1, 6 });

            Assert.AreEqual(expected, actual);
        }
    }
}
