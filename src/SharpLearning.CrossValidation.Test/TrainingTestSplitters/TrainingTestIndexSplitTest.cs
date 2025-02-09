using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingTestSplitters;

[TestClass]
public class TrainingTestIndexSplitTest
{
    [TestMethod]
    public void TrainingTestIndexSplit_Equals()
    {
        var sut = new TrainingTestIndexSplit([1, 2], [3, 4]);

        var equal = new TrainingTestIndexSplit([1, 2], [3, 4]);
        var notEqual = new TrainingTestIndexSplit([3, 4], [1, 2]);

        Assert.IsTrue(sut.Equals(equal));
        Assert.IsFalse(sut.Equals(notEqual));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TrainingTestIndexSplit_TrainingIndices_Is_Null()
    {
        new TrainingTestIndexSplit(null, [3, 4]);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TrainingTestIndexSplit_ValidationIndices_Is_Null()
    {
        new TrainingTestIndexSplit([3, 4], null);
    }
}
