using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Arithmetic;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Test.Matrices;

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
        expected.Map(() => 1.0);

        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MatrixAdd_AddInPlace_2()
    {
        var m = new F64Matrix(4, 3);
        m.Map(() => 1.0);
        var v = new double[] { 1, 1, 1, 1 };

        var actual = new F64Matrix(4, 3);
        MatrixAdd.AddF64(m, v, actual);

        var expected = new F64Matrix(4, 3);
        expected.Map(() => 2.0);

        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MatrixAdd_Add_Vectors()
    {
        var v1 = new double[] { 2, 3, 5, 10 };
        var v2 = new double[] { 1, 1, 1, 1 };

        var actual = v1.Add(v2);
        var expected = new double[] { 3, 4, 6, 11 };

        Assert.AreEqual(expected.Length, actual.Length);

        for (var i = 0; i < actual.Length; i++)
        {
            Assert.AreEqual(expected[i], actual[i], 0.1);
        }
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void MatrixAdd_Add_Vectors_Different_Lengths()
    {
        var v1 = new double[] { 2, 3, 5, 10 };
        var v2 = new double[] { 1, 1 };

        v1.Add(v2);
    }

}
