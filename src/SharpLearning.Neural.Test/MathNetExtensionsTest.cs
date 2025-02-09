using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace SharpLearning.Neural.Test;

[TestClass]
public class MathNetExtensionsTest
{
    [TestMethod]
    public void MathNetExtensions_AddRowWise()
    {
        var matrix = Matrix<float>.Build.Dense(2, 3);
        var vector = Vector<float>.Build.Dense(new float[] { 1f, 2f, 3f });
        var actual = Matrix<float>.Build.Dense(2, 3);

        matrix.AddRowWise(vector, actual);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));
        Trace.WriteLine(actual.ToString());

        var expected = Matrix<float>.Build.Dense(2, 3, new float[] { 1, 1, 2, 2, 3, 3 });
        Assert.AreEqual(expected.ToString(), actual.ToString());
    }

    [TestMethod]
    public void MathNetExtensions_SubtractRowWise()
    {
        var matrix = Matrix<float>.Build.Dense(2, 3);
        var vector = Vector<float>.Build.Dense(new float[] { 1f, 2f, 3f });
        var actual = Matrix<float>.Build.Dense(2, 3);

        matrix.SubtractRowWise(vector, actual);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));
        Trace.WriteLine(actual.ToString());

        var expected = Matrix<float>.Build.Dense(2, 3, new float[] { -1, -1, -2, -2, -3, -3 });
        Assert.AreEqual(expected.ToString(), actual.ToString());
    }

    [TestMethod]
    public void MathNetExtensions_AddColumnWise()
    {
        var matrix = Matrix<float>.Build.Dense(2, 3);
        var vector = Vector<float>.Build.Dense(new float[] { 1f, 2f });
        var actual = Matrix<float>.Build.Dense(2, 3);

        matrix.AddColumnWise(vector, actual);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));
        Trace.WriteLine(actual.ToString());

        var expected = Matrix<float>.Build.Dense(2, 3, new float[] { 1, 2, 1, 2, 1, 2 });
        Assert.AreEqual(expected.ToString(), actual.ToString());
    }

    [TestMethod]
    public void MathNetExtensions_Multiply()
    {
        var matrix = Matrix<float>.Build.Dense(2, 3, 1);
        var vector = Vector<float>.Build.Dense(new float[] { 1f, 0f, 1f });
        var actual = Matrix<float>.Build.Dense(2, 3);

        matrix.Multiply(vector, actual);

        Trace.WriteLine(string.Join(", ", actual.ToColumnMajorArray()));
        Trace.WriteLine(actual.ToString());

        var expected = Matrix<float>.Build.Dense(2, 3, new float[] { 1, 1, 0, 0, 1, 1 });
        Assert.AreEqual(expected.ToString(), actual.ToString());
    }

    [TestMethod]
    public void MathNetExtensions_ColumnWiseMean()
    {
        var matrix = Matrix<float>.Build.Dense(3, 3, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        var actual = Vector<float>.Build.Dense(3);

        matrix.ColumnWiseMean(actual);

        Trace.WriteLine(string.Join(", ", actual));
        Trace.WriteLine(matrix.ToString());

        var expected = Vector<float>.Build.Dense(new float[] { 2, 5, 8 });
        Assert.AreEqual(expected.ToString(), actual.ToString());
    }

    [TestMethod]
    public void MathNetExtensions_ColumnWiseMean_2()
    {
        var matrix = Matrix<float>.Build.Dense(4, 2, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var actual = Vector<float>.Build.Dense(3);

        matrix.ColumnWiseMean(actual);

        Trace.WriteLine(string.Join(", ", actual));
        Trace.WriteLine(matrix.ToString());

        var expected = Vector<float>.Build.Dense(new float[] { 2.5f, 6.5f, 0f });
        Assert.AreEqual(expected.ToString(), actual.ToString());
    }

    [TestMethod]
    public void MathNetExtensions_ColumnWiseMean_SumColumns()
    {
        var matrix = Matrix<float>.Build.Dense(4, 2, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var actual = Vector<float>.Build.Dense(2);

        matrix.SumColumns(actual);

        var expected = Vector<float>.Build.Dense(new float[] { 10, 26 });
        Assert.AreEqual(expected.ToString(), actual.ToString());
    }

    [TestMethod]
    public void MathNetExtensions_ColumnWiseMean_SumRows()
    {
        var matrix = Matrix<float>.Build.Dense(4, 2, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        var actual = Vector<float>.Build.Dense(4);

        matrix.SumRows(actual);

        var expected = Vector<float>.Build.Dense(new float[] { 6, 8, 10, 12 });
        Assert.AreEqual(expected.ToString(), actual.ToString());
    }

    [TestMethod]
    public void MathNetExtensions_Matrix_Data()
    {
        var expected = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var matrix = Matrix<float>.Build.Dense(4, 2, expected);
        var actual = matrix.Data();

        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MathNetExtensions_Matrix_Data_Modify()
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var matrix = Matrix<float>.Build.Dense(4, 2, input);

        var changeIndex = 2;
        var value = 666;

        var data = matrix.Data();
        data[changeIndex] = value;

        var expected = new float[] { 1, 2, value, 4, 5, 6, 7, 8 };
        var actual = matrix.Data();

        CollectionAssert.AreEqual(expected, actual);
        Assert.AreEqual(value, matrix[changeIndex, 0]);
    }

    [TestMethod]
    public void MathNetExtensions_Vector_Data()
    {
        var expected = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var vector = Vector<float>.Build.Dense(expected);
        var actual = vector.Data();

        Assert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void MathNetExtensions_Vector_Data_Modify()
    {
        var input = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var vector = Vector<float>.Build.Dense(input);

        var changeIndex = 2;
        var value = 666;

        var data = vector.Data();
        data[changeIndex] = value;

        var expected = new float[] { 1, 2, value, 4, 5, 6, 7, 8 };
        var actual = vector.Data();

        CollectionAssert.AreEqual(expected, actual);
        Assert.AreEqual(value, vector[changeIndex]);
    }


    [TestMethod]
    public void MathNetExtensions_Matri_Row()
    {
        var data = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var matrix = Matrix<float>.Build.Dense(4, 2, data);
        var actual = new float[2];
        matrix.Row(1, actual);

        var expected = new float[] { 2, 6 };
        CollectionAssert.AreEqual(expected, actual);
    }
}
