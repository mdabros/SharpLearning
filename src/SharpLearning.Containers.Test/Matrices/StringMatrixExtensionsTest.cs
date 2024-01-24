using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Test.Matrices
{
    [TestClass]
    public class StringMatrixExtensionsTest
    {
        readonly string[] m_inputData = new string[] { "1", "2", "3", "4", "5", "6" };
        readonly string[] m_combineDataCols = new string[] { "1", "2", "3", "1", "2", "3", "4", "5", "6", "4", "5", "6" };
        readonly string[] m_combineDataRows = new string[] { "1", "2", "3", "4", "5", "6", "1", "2", "3", "4", "5", "6" };

        [TestMethod]
        public void StringMatrixExtensions_ToF64Matrix()
        {
            var stringMatrix = new StringMatrix(m_inputData, 2, 3);
            var actual = stringMatrix.ToF64Matrix();

            var expected = new F64Matrix(m_inputData.Select(v =>
                FloatingPointConversion.ToF64(v)).ToArray(), 2, 3);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_Map()
        {
            var matrix = new StringMatrix(m_inputData.ToArray(), 2, 3);
            matrix.Map(() => "10");

            var expected = Enumerable.Range(0, matrix.Data().Length).Select(v => "10").ToArray();
            CollectionAssert.AreEqual(expected, matrix.Data());
        }

        [TestMethod]
        public void StringMatrixExtensions_Map2()
        {
            var matrix = new StringMatrix(m_inputData.ToArray(), 2, 3);
            matrix.Map(() => "10");
            matrix.Map(v => v + "1");

            var expected = Enumerable.Range(0, matrix.Data().Length).Select(v => "101").ToArray();
            CollectionAssert.AreEqual(expected, matrix.Data());
        }

        [TestMethod]
        public void StringMatrixExtensions_CombineStringMatrices_Cols()
        {
            var matrix1 = new StringMatrix(m_inputData, 2, 3);
            var matrix2 = new StringMatrix(m_inputData, 2, 3);

            var actual = matrix1.CombineCols(matrix2);

            Assert.AreEqual(new StringMatrix(m_combineDataCols, 2, 6), actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_CombineStringMatrices_Rows()
        {
            var matrix1 = new StringMatrix(m_inputData, 2, 3);
            var matrix2 = new StringMatrix(m_inputData, 2, 3);

            var actual = matrix1.CombineRows(matrix2);

            Assert.AreEqual(new StringMatrix(m_combineDataRows, 4, 3), actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_CombineStringMatrixAndVector()
        {
            var matrix = new StringMatrix(m_inputData, 2, 3);
            var vector = new string[] { "3", "6" };

            var expected = new StringMatrix(new string[] {"1", "2", "3", "3",
                                                          "4", "5", "6", "6"}, 2, 4);
            var actual = matrix.CombineCols(vector);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void StringMatrixExtensions_CombineStringVectorAndMatrixAnd()
        {
            var matrix = new StringMatrix(m_inputData, 2, 3);
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
            var expected = new StringMatrix(new string[] { "1", "1", "2", "2", "3", "3", "4", "4" }, 4, 2);

            Assert.AreEqual(expected, actual);
        }
    }
}
