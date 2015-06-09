using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Metrics.Test.Classification
{
    [TestClass]
    public class ClassificationMatrixTest
    {
        [TestMethod]
        public void ClassificationMatrix_ConfusionMatrix()
        {
            var predictions = new double[] { 0, 1, 2 };
            var targets = new double[] { 0, 1, 1 };
            var uniqueTargets = predictions.Distinct().ToList();

            var actual = ClassificationMatrix.ConfusionMatrix(uniqueTargets, targets, predictions);

            var expected = new int[][] { new int[] { 1, 0, 0 }, new int[] { 0, 1, 1 }, new int[] { 0, 0, 0 } };

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                CollectionAssert.AreEqual(expected[i], actual[i]);
            }
        }

        [TestMethod]
        public void ClassificationMatrix_ErrorMatrix()
        {
            var uniqueTargets = new List<double> {0, 1, 2};
            var confusionmatrix = new int[][] { new int[] { 1, 0, 0 }, new int[] { 0, 1, 1 }, new int[] { 0, 0, 0 } };

            var actual = ClassificationMatrix.ErrorMatrix(uniqueTargets, confusionmatrix);

            var expected = new double[][] { new double[] { 1, 0, 0 }, new double[] { 0, .5, .5 }, new double[] { 0, 0, 0 } };

            Assert.AreEqual(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                CollectionAssert.AreEqual(expected[i], actual[i]);
            }
        }

    }
}
