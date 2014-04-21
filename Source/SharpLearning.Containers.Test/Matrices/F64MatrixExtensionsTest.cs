using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using SharpLearning.Containers.Matrices;
using System.Linq;

namespace SharpLearning.Containers.Test.Matrices
{
    [TestClass]
    public class F64MatrixExtensionsTest
    {
        readonly double[] InputData = new double[] { 1, 2, 3, 4, 5, 6 };
        readonly double[] CombineData = new double[] { 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6 };

        [TestMethod]
        public void F64MatrixExtensions_ToStringMatrix()
        {
            var matrix = new F64Matrix(InputData, 2, 3);
            var actual = matrix.ToStringMatrix();

            var expected = new StringMatrix(InputData.Select(v => FloatingPointConversion.ToString(v)).ToArray(), 2, 3);
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_CombineF64Matrices()
        {
            var matrix1 = new F64Matrix(InputData, 2, 3);
            var matrix2 = new F64Matrix(InputData, 2, 3);

            var actual = matrix1.CombineCols(matrix2);

            Assert.AreEqual(new F64Matrix(CombineData, 2, 6), actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_CombineF64MatrixAndVector()
        {
            var matrix = new F64Matrix(InputData, 2, 3);
            var vector = new double[] { 3, 6 };

            var expected = new F64Matrix(new double[] {1, 2, 3, 3,
                                                       4, 5, 6, 6}, 2, 4);
            var actual = matrix.CombineCols(vector);
            
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_VectorAndVector()
        {
            var v1 = new double[] { 1, 2, 3, 4 };
            var v2 = new double[] { 1, 2, 3, 4 };

            var actual = v1.CombineCols(v2);
            Assert.AreEqual(new F64Matrix(new double[] { 1, 1, 2, 2, 3, 3, 4, 4}, 4, 2), actual);
        }
    }
}
