using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingTestSplitters;

[TestClass]
public class NoShuffleTrainingTestIndexSplitterTest
{
    [TestMethod]
    public void NoShuffleTrainingValidationIndexSplitter_Split()
    {
        var sut = new NoShuffleTrainingTestIndexSplitter<double>(0.8);

        var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        var actual = sut.Split(targets);
        var expected = new TrainingTestIndexSplit([0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9]);

        Assert.AreEqual(expected, actual);
    }
}
