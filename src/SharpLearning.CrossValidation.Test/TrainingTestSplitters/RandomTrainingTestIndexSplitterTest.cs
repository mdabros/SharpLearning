using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingTestSplitters
{
    [TestClass]
    public class RandomTrainingTestIndexSplitterTest
    {
        [TestMethod]
        public void RandomTrainingTestIndexSplitter_Split()
        {
            var sut = new RandomTrainingTestIndexSplitter<double>(0.8, 42);

            var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

            var actual = sut.Split(targets);
            var expected = new TrainingTestIndexSplit(new int[] { 9, 0, 4, 2, 5, 7, 3, 8 },
                new int[] { 1, 6 });

            Assert.AreEqual(expected, actual);
        }
    }
}
