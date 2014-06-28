using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Shufflers;
using System.Linq;

namespace SharpLearning.CrossValidation.Test.Shufflers
{
    [TestClass]
    public class RandomCrossValidationShufflerTest
    {
        [TestMethod]
        public void RandomCrossValidationShuffler_Shuffle()
        {
            var actual = Enumerable.Range(0, 10).ToArray();
            var targets = new double[] { 0, 1, 2, 3 ,4 ,5 ,6 ,7 ,8 ,9 };
            var sut = new RandomCrossValidationShuffler<double>(42);

            sut.Shuffle(actual, targets, 3);

            var expected = new int[] { 9, 0, 4, 2, 5, 7, 3, 8, 1, 6 };
            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
