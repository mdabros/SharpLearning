using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using System.Collections.Generic;
using System.Linq;

namespace SharpLearning.Containers.Test.Matrices
{
    [TestClass]
    public class StringMatrixExtensionsTest
    {
        readonly string[] InputData = new string[] { "1", "2", "3", "4", "5", "6" };
        readonly string[] CombineDataCols = new string[] { "1", "2", "3", "1", "2", "3", "4", "5", "6", "4", "5", "6" };
        readonly string[] CombineDataRows = new string[] { "1", "2", "3", "4", "5", "6", "1", "2", "3", "4", "5", "6" };

        [TestMethod]
        public void StringMatrixExtensions_ToF64Matrix()
        {
            var stringMatrix = new StringMatrix(InputData, 2, 3);
            var actual = stringMatrix.ToF64Matrix();

            Assert.AreEqual(new F64Matrix(InputData.Select(v => FloatingPointConversion.ToF64(v)).ToArray(), 2, 3), actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_CombineStringMatrices_Cols()
        {
            var matrix1 = new StringMatrix(InputData,2, 3);
            var matrix2 = new StringMatrix(InputData,2, 3);

            var actual = matrix1.CombineCols(matrix2);

            Assert.AreEqual(new StringMatrix(CombineDataCols, 2, 6), actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_CombineStringMatrices_Rows()
        {
            var matrix1 = new StringMatrix(InputData, 2, 3);
            var matrix2 = new StringMatrix(InputData, 2, 3);

            var actual = matrix1.CombineRows(matrix2);

            Assert.AreEqual(new StringMatrix(CombineDataRows, 4, 3), actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_CombineStringMatrixAndVector()
        {
            var matrix = new StringMatrix(InputData, 2, 3);
            var vector = new string[] { "3", "6" };

            var expected = new StringMatrix(new string[] {"1", "2", "3", "3",
                                                          "4", "5", "6", "6"}, 2, 4);
            var actual = matrix.CombineCols(vector);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_CombineStringVectorAndMatrixAnd()
        {
            var matrix = new StringMatrix(InputData, 2, 3);
            var vector = new string[] { "3", "6" };

            var expected = new StringMatrix(new string[] {"3", "1", "2", "3",
                                                          "6", "4", "5", "6", }, 2, 4);
            var actual = vector.CombineCols(matrix);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_VectorAndVector()
        {
            var v1 = new string[] { "1", "2", "3", "4" };
            var v2 = new string[] { "1", "2", "3", "4" };

            var actual = v1.CombineCols(v2);
            Assert.AreEqual(new StringMatrix(new string[] { "1", "1", "2", "2", "3", "3", "4", "4" }, 4, 2), actual);
        }
    }
}
