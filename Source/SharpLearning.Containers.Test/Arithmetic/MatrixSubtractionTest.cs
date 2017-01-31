using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Arithmetic;
using SharpLearning.Containers.Matrices;
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
        public void MatrixSubtraction_SubtractF64Matrix_1()
        {
            var m1 = new F64Matrix(4, 3);
            var m2 = new F64Matrix(4, 3);
            m2.Map(() => 1.0);

            var actual = new F64Matrix(4, 3);
            MatrixSubtraction.SubtractF64(m1, m2, actual);

            var expected = new F64Matrix(4, 3);
            expected.Map(() => -1.0);
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void MatrixSubtraction_SubtractF64Matrix_2()
        {
            var m1 = new F64Matrix(4, 3);
            m1.Map(() => 5);

            var m2 = new F64Matrix(4, 3);
            m2.Map(() => 1.0);

            var actual = new F64Matrix(4, 3);
            MatrixSubtraction.SubtractF64(m1, m2, actual);

            var expected = new F64Matrix(4, 3);
            expected.Map(() => 4.0);
            Assert.AreEqual(expected, actual);
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
