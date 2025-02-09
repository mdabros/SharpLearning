using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.TrainingTestSplitters;

namespace SharpLearning.CrossValidation.Test.TrainingTestSplitters;

[TestClass]
public class TrainingTestIndexSplitterExtensionsTest
{
    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TrainingTestIndexSplitterExtensions_SplitSet_Observations_Targets_Row_Differ()
    {
        var splitter = new RandomTrainingTestIndexSplitter<double>(0.6, 32);
        splitter.SplitSet(new F64Matrix(10, 2), new double[8]);
    }

    [TestMethod]
    public void TrainingTestIndexSplitterExtensions_SplitSet()
    {
        var observations = new F64Matrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10, 1);
        var targets = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var splitter = new NoShuffleTrainingTestIndexSplitter<double>(0.6);

        var actual = splitter.SplitSet(observations, targets);

        var trainingIndices = Enumerable.Range(0, 6).ToArray();
        var testIndices = Enumerable.Range(6, 4).ToArray();

        var expected = new TrainingTestSetSplit(
            new ObservationTargetSet((F64Matrix)observations.Rows(trainingIndices),
                targets.GetIndices(trainingIndices)),
            new ObservationTargetSet((F64Matrix)observations.Rows(testIndices),
                targets.GetIndices(testIndices)));

        Assert.AreEqual(expected, actual);
    }
}
