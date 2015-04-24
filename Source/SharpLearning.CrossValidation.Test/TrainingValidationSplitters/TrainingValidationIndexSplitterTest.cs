using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Samplers;
using SharpLearning.CrossValidation.TrainingValidationSplitters;
using System;

namespace SharpLearning.CrossValidation.Test.TrainingValidationSplitters
{
    [TestClass]
    public class TrainingValidationIndexSplitterTest
    {
        [TestMethod]
        public void TrainingValidationIndexSplitter_Split()
        {
            var sut = new TrainingValidationIndexSplitter<double>(
                new NoShuffleIndexSampler<double>(), 0.8);
            
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            var actual = sut.Split(targets);
            var expected = new TrainingValidationIndexSplit(new int[] { 0, 1, 2, 3, 4, 5, 6, 7 },
                new int[] { 8, 9 });

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TrainingValidationIndexSplitter_Training_Percentage_Too_Low()
        {
            var sut = new TrainingValidationIndexSplitter<double>(
                new NoShuffleIndexSampler<double>(), 0.0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TrainingValidationIndexSplitter_Training_Percentage_Too_High()
        {
            var sut = new TrainingValidationIndexSplitter<double>(
                new NoShuffleIndexSampler<double>(), 1.0);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void TrainingValidationIndexSplitter_Shuffler_Is_Null()
        {
            var sut = new TrainingValidationIndexSplitter<double>(
                null, 0.8);
        }
    }
}
