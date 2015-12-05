using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Arithmetic;

namespace SharpLearning.Containers.Test.Matrices
{
    [TestClass]
    public class MatrixAddTest
    {
        [TestMethod]
        public void MatrixAdd_AddInPlace_1()
        {
            var m = new F64Matrix(4, 3);
            var v = new double[] { 1, 1, 1, 1 };

            var actual = new F64Matrix(4, 3);
            MatrixAdd.AddF64(m, v, actual);

            var expected = new F64Matrix(4, 3);
            expected.Initialize(() => 1.0);

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void MatrixAdd_AddInPlace_2()
        {
            var m = new F64Matrix(4, 3);
            m.Initialize(() => 1.0);
            var v = new double[] { 1, 1, 1, 1 };

            var actual = new F64Matrix(4, 3);
            MatrixAdd.AddF64(m, v, actual);

            var expected = new F64Matrix(4, 3);
            expected.Initialize(() => 2.0);

            Assert.AreEqual(expected, actual);
        }
    }
}
