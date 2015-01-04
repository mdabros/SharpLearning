using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Shufflers;
using System.Linq;

namespace SharpLearning.CrossValidation.Test.Shufflers
{
    [TestClass]
    public class NoShuffleCrossValidationShufflerTest
    {
        [TestMethod]
        public void NoShuffleCrossValidationShuffler_Shuffle()
        {
            var actual = Enumerable.Range(0, 10).ToArray();
            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var sut = new NoShuffleCrossValidationShuffler<double>();

            sut.Shuffle(actual, targets, 3);

            var expected = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
