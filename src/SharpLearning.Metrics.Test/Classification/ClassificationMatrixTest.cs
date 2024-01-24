using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;

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
            var expected = new int[,] { { 1, 0, 0 }, { 0, 1, 1 }, { 0, 0, 0 } };

            AssertAreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationMatrix_ErrorMatrix()
        {
            var uniqueTargets = new List<double> { 0, 1, 2 };
            var confusionmatrix = new int[,] { { 1, 0, 0 }, { 0, 1, 1 }, { 0, 0, 0 } };

            var actual = ClassificationMatrix.ErrorMatrix(uniqueTargets, confusionmatrix);
            var expected = new double[,] { { 1, 0, 0 }, { 0, .5, .5 }, { 0, 0, 0 } };

            AssertAreEqual(expected, actual);
        }

        static void AssertAreEqual<T>(T[,] expected, T[,] actual)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            for (int r = 0; r < expected.GetLength(0); r++)
            {
                for (int c = 0; c < expected.GetLength(1); c++)
                {
                    Assert.AreEqual(expected[r, c], actual[r, c]);
                }
            }
        }
    }
}
