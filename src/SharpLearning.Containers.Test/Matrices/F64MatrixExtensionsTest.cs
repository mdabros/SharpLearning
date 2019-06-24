using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Test.Matrices
{
    [TestClass]
    public class F64MatrixExtensionsTest
    {
        readonly double[] m_inputData = new double[] { 1, 2, 3, 4, 5, 6 };
        readonly double[] m_combineDataCol = new double[] { 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6 };
        readonly double[] m_combineDataRows = new double[] { 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 };

        [TestMethod]
        public void F64MatrixExtensions_Clear()
        {
            var matrix = new F64Matrix(m_inputData.ToArray(), 2, 3);
            matrix.Clear();

            CollectionAssert.AreEqual(new double[2 * 3], matrix.Data());
        }

        [TestMethod]
        public void F64MatrixExtensions_Map()
        {
            var matrix = new F64Matrix(m_inputData.ToArray(), 2, 3);
            matrix.Map(() => 10);

            var expected = Enumerable.Range(0, matrix.Data().Length).Select(v => 10.0).ToArray();
            CollectionAssert.AreEqual(expected, matrix.Data());
        }
        
        [TestMethod]
        public void F64MatrixExtensions_Map2()
        {
            var matrix = new F64Matrix(m_inputData.ToArray(), 2, 3);
            matrix.Map(() => 10);
            matrix.Map(v => v + 1);

            var expected = Enumerable.Range(0, matrix.Data().Length).Select(v => 11.0).ToArray();
            CollectionAssert.AreEqual(expected, matrix.Data());
        }
        
        [TestMethod]
        public void F64MatrixExtensions_ToStringMatrix()
        {
            var matrix = new F64Matrix(m_inputData, 2, 3);
            var actual = matrix.ToStringMatrix();

            var expected = new StringMatrix(m_inputData.Select(v => FloatingPointConversion.ToString(v)).ToArray(), 2, 3);
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_CombineF64Matrices()
        {
            var matrix1 = new F64Matrix(m_inputData, 2, 3);
            var matrix2 = new F64Matrix(m_inputData, 2, 3);

            var actual = matrix1.CombineCols(matrix2);

            Assert.AreEqual(new F64Matrix(m_combineDataCol, 2, 6), actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_CombineF64MatrixAndVector()
        {
            var matrix = new F64Matrix(m_inputData, 2, 3);
            var vector = new double[] { 3, 6 };

            var expected = new F64Matrix(new double[] {1, 2, 3, 3,
                                                       4, 5, 6, 6}, 2, 4);
            var actual = matrix.CombineCols(vector);
            
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_CombineVectorAndF64Matrix()
        {
            var matrix = new F64Matrix(m_inputData, 2, 3);
            var vector = new double[] { 3, 6 };

            var expected = new F64Matrix(new double[] {3, 1, 2, 3,
                                                       6, 4, 5, 6 }, 2, 4);
            var actual = vector.CombineCols(matrix);

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

        [TestMethod]
        public void F64MatrixExtensions_CombineRows_VectorAndVector()
        {
            var v1 = new double[] { 1, 2, 3, 4 };
            var v2 = new double[] { 1, 2, 3, 4 };

            var actual = v1.CombineRows(v2);
            Assert.AreEqual(new F64Matrix(new double[] { 1, 2, 3, 4, 1, 2, 3, 4 }, 2, 4), actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_CombineRows_F64MatrixAndVector()
        {
            var matrix = new F64Matrix(m_inputData, 2, 3);
            var vector = new double[] { 3, 6, 7 };

            var expected = new F64Matrix(new double[] {1, 2, 3, 
                                                       4, 5, 6, 
                                                       3, 6, 7}, 3, 3);
            var actual = matrix.CombineRows(vector);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_CombineRows_VectorAndF64Matrix()
        {
            var matrix = new F64Matrix(m_inputData, 2, 3);
            var vector = new double[] { 3, 6, 7 };

            var expected = new F64Matrix(new double[] {3, 6, 7,
                                                       1, 2, 3, 
                                                       4, 5, 6 
                                                       }, 3, 3);
            var actual = vector.CombineRows(matrix);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void F64MatrixExtensions_CombineRows_F64Matrices()
        {
            var matrix1 = new F64Matrix(m_inputData, 2, 3);
            var matrix2 = new F64Matrix(m_inputData, 2, 3);

            var actual = matrix1.CombineRows(matrix2);

            Assert.AreEqual(new F64Matrix(m_combineDataRows, 4, 3), actual);
        }
    }
}
