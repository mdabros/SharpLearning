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
        public void MatrixAdd_AddInPlace()
        {
            var m = new F64Matrix(4, 3);
            var v = new double[] { 1, 1, 1, 1 };

            MatrixAdd.AddInPlace(m, v);

            var expected = new F64Matrix(4, 3);
            expected.Initialize(() => 1.0);

            Assert.AreEqual(expected, m);
        }
    }
}
