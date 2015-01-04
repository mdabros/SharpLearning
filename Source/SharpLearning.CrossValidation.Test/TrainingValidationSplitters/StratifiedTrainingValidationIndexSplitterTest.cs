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
            var sut = new StratifiedTrainingValidationIndexSplitter<double>(0.8, 42);

            var targets = new double[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 };

            var actual = sut.Split(targets);
            var expected = new TrainingValidationIndexSplit(new int[] { 9, 0, 5, 4, 7, 2, 8, 3 },
                new int[] { 6, 1 });

            Assert.AreEqual(expected, actual);
        }
    }
}
