using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Shufflers;
using System;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.CrossValidation.Test.Shufflers
{
    [TestClass]
    public class StratifyCrossValidationShufflerTest
    {
        [TestMethod]
        public void StratifyCrossValidationShuffler_Shuffle_Even()
        {
            var targets = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var actual = Enumerable.Range(0, targets.Length).ToArray();
            var sut = new StratifyCrossValidationShuffler<int>(42);

            sut.Shuffle(actual, targets, 2);

            var expected = new int[] { 8, 6, 0, 1, 5, 16, 12, 17, 10, 21, 18, 7, 4, 3, 9, 2, 15, 13, 11, 14, 19, 20 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void StratifyCrossValidationShuffler_Shuffle_Uneven()
        {
            var targets = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var actual = Enumerable.Range(0, targets.Length).ToArray();
            var sut = new StratifyCrossValidationShuffler<int>(42);

            sut.Shuffle(actual, targets, 3);
            Write(actual);

            var expected = new int[] { 8, 6, 0, 16, 12, 21, 1, 5, 7, 17, 10, 18, 4, 3, 9, 2, 15, 13, 11, 14, 19, 20 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void StratifyCrossValidationShuffler_Shuffle_Too_Many_Folds()
        {
            var targets = new int[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3 };
            var actual = Enumerable.Range(0, targets.Length).ToArray();
            var sut = new StratifyCrossValidationShuffler<int>(42);

            sut.Shuffle(actual, targets, 10);

            var expected = new int[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, };
            CollectionAssert.AreEqual(expected, actual);
        }

        void Write(int[] indices)
        {
            var output = "new int[] {";
            foreach (var index in indices)
            {
                output += index + ", ";
            }
            output += "};";
            Trace.WriteLine(output);
        }
    }
}
