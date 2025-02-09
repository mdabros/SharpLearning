using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingTestSplitters;

[TestClass]
public class TrainingTestSetSplitTest
{
    [TestMethod]
    public void TrainingTestSetSplit_Equals()
    {
        var trainingObservations1 = new F64Matrix([1, 2, 3, 4], 2, 2);
        var trainingObservations2 = new F64Matrix([4, 3, 2, 1], 2, 2);

        var trainingTargets1 = new double[] { 1, 2 };
        var trainingTargets2 = new double[] { 2, 1 };

        var testObservations1 = new F64Matrix([1, 2, 3, 4], 2, 2);
        var testObservations2 = new F64Matrix([4, 3, 2, 1], 2, 2);

        var testTargets1 = new double[] { 1, 2 };
        var testTargets2 = new double[] { 2, 1 };


        var sut = new TrainingTestSetSplit(trainingObservations1, trainingTargets1, testObservations1, testTargets1);
        var equal = new TrainingTestSetSplit(trainingObservations1, trainingTargets1, testObservations1, testTargets1);
        var notEqual1 = new TrainingTestSetSplit(trainingObservations2, trainingTargets1, testObservations1, testTargets1);
        var notEqual2 = new TrainingTestSetSplit(trainingObservations1, trainingTargets2, testObservations1, testTargets1);
        var notEqual3 = new TrainingTestSetSplit(trainingObservations1, trainingTargets1, testObservations2, testTargets1);
        var notEqual4 = new TrainingTestSetSplit(trainingObservations1, trainingTargets1, testObservations1, testTargets2);

        Assert.AreEqual(sut, equal);
        Assert.AreNotEqual(sut, notEqual1);
        Assert.AreNotEqual(sut, notEqual2);
        Assert.AreNotEqual(sut, notEqual3);
        Assert.AreNotEqual(sut, notEqual4);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TrainingTestSetSplit_TrainingSet_Null()
    {
        new TrainingTestSetSplit(null,
            new ObservationTargetSet(new F64Matrix([1, 2, 3, 4], 2, 2), [1, 2]));
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void ObservationTargetSet_TestSet_Null()
    {
        new TrainingTestSetSplit(new ObservationTargetSet(new F64Matrix([1, 2, 3, 4], 2, 2), [1, 2]),
            null);
    }
}
