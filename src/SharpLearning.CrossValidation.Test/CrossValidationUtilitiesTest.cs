using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.Test;

[TestClass]
public class CrossValidationUtilitiesTest
{
    [TestMethod]
    public void CrossValidationUtilities_GetKFoldCrossValidationIndexSets()
    {
        var targets = new double[] { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
        var sampler = new StratifiedIndexSampler<double>(seed: 242);
        var actuals = CrossValidationUtilities.GetKFoldCrossValidationIndexSets(sampler,
            foldCount: 4, targets: targets);

        TraceIndexSets(actuals);

        var expecteds = new List<(int[] trainingIndices, int[] validationIndices)>
        {
            ([0, 1, 3, 4, 5, 7, 9, 10, 11], [6, 8, 2]),
            ([0, 2, 3, 4, 6, 7, 8, 9, 10 ], [1, 11, 5]),
            ([0, 1, 2, 4, 5, 6, 8, 9, 11 ], [7, 3, 10]),
            ([1, 2, 3, 5, 6, 7, 8, 10, 11], [0, 4, 9]),
        };

        Assert.AreEqual(expecteds.Count, actuals.Count);
        for (var i = 0; i < expecteds.Count; i++)
        {
            var expected = expecteds[i];
            var actual = actuals[i];
            CollectionAssert.AreEqual(expected.trainingIndices, actual.trainingIndices);
            CollectionAssert.AreEqual(expected.validationIndices, actual.validationIndices);
        }
    }

    [TestMethod]
    public void CrossValidationUtilities_GetKFoldCrossValidationIndexSets_Indices()
    {
        var targets = new double[] { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 };
        var indices = new int[] { 0, 1, 2, 3, 4, 5, 6 };
        var sampler = new StratifiedIndexSampler<double>(seed: 242);
        var actuals = CrossValidationUtilities.GetKFoldCrossValidationIndexSets(sampler,
            foldCount: 2, targets: targets, indices: indices);

        TraceIndexSets(actuals);

        var expecteds = new List<(int[] trainingIndices, int[] validationIndices)>
        {
            // Sets contains values from the indices array only.
            (new int[] { 1, 3, 4, 5 }, new int[] { 2, 6, 0 }),
            (new int[] { 0, 2, 6 }, new int[] { 1, 3, 4, 5 }),
        };

        Assert.AreEqual(expecteds.Count, actuals.Count);
        for (var i = 0; i < expecteds.Count; i++)
        {
            var expected = expecteds[i];
            var actual = actuals[i];
            CollectionAssert.AreEqual(expected.trainingIndices, actual.trainingIndices);
            CollectionAssert.AreEqual(expected.validationIndices, actual.validationIndices);
        }
    }

    [TestMethod]
    public void CrossValidationUtilities_GetKFoldCrossValidationIndexSets_Handle_Remainder()
    {
        var targets = new double[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3 };
        var sampler = new StratifiedIndexSampler<double>(seed: 242);
        var actuals = CrossValidationUtilities.GetKFoldCrossValidationIndexSets(sampler,
            foldCount: 4, targets: targets);

        var expecteds = new List<(int[] trainingIndices, int[] validationIndices)>
        {
            (new int[] { 0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 14 }, new int[] { 10, 4, 8 }),
            (new int[] { 0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13 }, new int[] { 2, 7, 14 }),
            (new int[] { 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14 }, new int[] { 5, 13, 0 }),
            // Handle remainder from target.length / foldsCount, 
            // by adding remaining indices to the last set
            (new int[] { 0, 2, 4, 5, 7, 8, 10, 13, 14 }, new int[] { 1, 3, 6, 9, 11, 12 }),
        };

        Assert.AreEqual(expecteds.Count, actuals.Count);
        for (var i = 0; i < expecteds.Count; i++)
        {
            var expected = expecteds[i];
            var actual = actuals[i];
            CollectionAssert.AreEqual(expected.trainingIndices, actual.trainingIndices);
            CollectionAssert.AreEqual(expected.validationIndices, actual.validationIndices);
        }
    }

    static void TraceIndexSets(IReadOnlyList<(int[] trainingIndices, int[] validationIndices)> indexSets)
    {
        const string separator = ", ";
        foreach (var set in indexSets)
        {
            Trace.WriteLine("(new int[] { " + string.Join(separator, set.trainingIndices) + " }, " +
                "new int[] { " + string.Join(separator, set.validationIndices) + " }),");
        }
    }
}
