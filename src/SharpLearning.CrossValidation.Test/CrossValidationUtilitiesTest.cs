using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.CrossValidation.Samplers;

namespace SharpLearning.CrossValidation.Test
{
    [TestClass]
    public class CrossValidationUtilitiesTest
    {
        [TestMethod]
        public void CrossValidationUtilities_GetCrossValidationIndexSets()
        {
            var targets = new double[] { 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3 };
            var sampler = new StratifiedIndexSampler<double>(seed: 242);
            var actuals = CrossValidationUtilities.GetCrossValidationIndexSets(sampler, 
                foldCount: 4, targets: targets);

            var expecteds = new List<(int[] trainingIndices, int[] validationIndices)>
            {
                (new int[] { 0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 14 }, new int[] { 10, 4, 8 }),
                (new int[] { 0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13 }, new int[] { 2, 7, 14 }),
                (new int[] { 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14 }, new int[] { 5, 13, 0 }),
                (new int[] { 0, 1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14 }, new int[] { 6, 11, 3 }),
            };

            Assert.AreEqual(expecteds.Count, actuals.Count);
            for (int i = 0; i < expecteds.Count; i++)
            {
                var expected = expecteds[i];
                var actual = actuals[i];
                CollectionAssert.AreEqual(expected.trainingIndices, actual.trainingIndices);
                CollectionAssert.AreEqual(expected.validationIndices, actual.validationIndices);
            }
        }

        void TraceIndexSets(List<(int[] trainingIndices, int[] validationIndices)> indexSets)
        {
            const string Separator = ", ";
            foreach (var set in indexSets)
            {
                Trace.WriteLine("(new int[] { " + string.Join(Separator, set.trainingIndices) + " }, " +
                    "new int[] { " + string.Join(Separator, set.validationIndices) + " }),");
            }
        }
    }
}
