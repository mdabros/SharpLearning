using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Arithmetic;
using System;

namespace SharpLearning.Containers.Test.Arithmetic
{
    [TestClass]
    public class MatrixSubtractionTest
    {
        [TestMethod]
        public void MatrixSubtraction_Subtract()
        {
            var v1 = new double[] { 1, 2, 3, 4, 5 };
            var v2 = new double[] { 5, 4, 3, 2, 1 };

            var actual = v1.Subtract(v2);
            var expected = new double[] { -4, -2, 0, 2, 4 };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void MatrixSubtraction_NonMatchingDimensions()
        {
            var v1 = new double[] { 1, 2, 3, 4, 5 };
            var v2 = new double[] { 5, 4 };

            var actual = v1.Subtract(v2);
        }
    }
}
