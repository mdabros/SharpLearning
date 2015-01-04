using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingValidationSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingValidationSplitters
{
    [TestClass]
    public class RandomTrainingValidationIndexSplitterTest
    {
        [TestMethod]
        public void RandomTrainingValidationIndexSplitter_Split()
        {
            var sut = new RandomTrainingValidationIndexSplitter<double>(0.8, 42);

            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            var actual = sut.Split(targets);
            var expected = new TrainingValidationIndexSplit(new int[] { 9, 0, 4, 2, 5, 7, 3, 8 },
                new int[] { 1, 6 });

            Assert.AreEqual(expected, actual);
        }
    }
}
