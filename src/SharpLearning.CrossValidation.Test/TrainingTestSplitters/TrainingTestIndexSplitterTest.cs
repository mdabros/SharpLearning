using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Samplers;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingTestSplitters;

[TestClass]
public class TrainingTestIndexSplitterTest
{
    [TestMethod]
    public void TrainingTestIndexSplitter_Split()
    {
        var sut = new TrainingTestIndexSplitter<double>(
            new NoShuffleIndexSampler<double>(), 0.8);

        var targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        var actual = sut.Split(targets);
        var expected = new TrainingTestIndexSplit([0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9]);

        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TrainingTestIndexSplitter_Training_Percentage_Too_Low()
    {
        var sut = new TrainingTestIndexSplitter<double>(
            new NoShuffleIndexSampler<double>(), 0.0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TrainingTestIndexSplitter_Training_Percentage_Too_High()
    {
        var sut = new TrainingTestIndexSplitter<double>(
            new NoShuffleIndexSampler<double>(), 1.0);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TrainingTestIndexSplitter_Shuffler_Is_Null()
    {
        var sut = new TrainingTestIndexSplitter<double>(
            null, 0.8);
    }
}
