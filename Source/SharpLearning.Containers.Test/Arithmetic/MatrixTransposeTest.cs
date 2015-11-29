using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Arithmetic;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Test.Arithmetic
{
    [TestClass]
    public class MatrixTransposeTest
    {
        [TestMethod]
        public void MatrixTranspose_Transpose()
        {
            var matrix = new F64Matrix(new double[6] { 1, 2, 3, 4, 5, 6 }, 3, 2);
            var transpose = matrix.Transpose();

            var expected = new F64Matrix(new double[6] { 1, 3, 5, 2, 4, 6 }, 2, 3);
            Assert.AreEqual(expected, transpose);
        }

        [TestMethod]
        public void MatrixTranspose_Transpose_Predefined()
        {
            var matrix = new F64Matrix(new double[6] { 1, 2, 3, 4, 5, 6 }, 3, 2);
            var transpose = new F64Matrix(2, 3);
            MatrixTranspose.TransposeF64(matrix, transpose);

            var expected = new F64Matrix(new double[6] { 1, 3, 5, 2, 4, 6 }, 2, 3);
            Assert.AreEqual(expected, transpose);
        }
    }
}
