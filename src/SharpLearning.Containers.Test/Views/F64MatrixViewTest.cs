using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;

namespace SharpLearning.Containers.Test.Views;

[TestClass]
public class F64MatrixViewTest
{
    [TestMethod]
    public unsafe void F64MatrixView_Indexing()
    {
        var matrix = Matrix();
        using var pinnedMatrix = matrix.GetPinnedPointer();
        var view = pinnedMatrix.View();
        AssertMatrixView(matrix, view);
    }

    [TestMethod]
    public unsafe void F64MatrixView_ColumnView()
    {
        var matrix = Matrix();
        using var pinnedMatrix = matrix.GetPinnedPointer();
        var view = pinnedMatrix.View();
        for (var i = 0; i < matrix.ColumnCount; i++)
        {
            AssertColumnView(matrix.Column(i), view.ColumnView(i));
        }
    }

    [TestMethod]
    public unsafe void F64MatrixView_SubView()
    {
        var matrix = Matrix();
        using var pinnedMatrix = matrix.GetPinnedPointer();
        var subView = pinnedMatrix.View().View(
            Interval2D.Create(Interval1D.Create(0, 2),
            Interval1D.Create(0, 3)));

        var subMatrix = matrix.Rows([0, 1]);
        AssertMatrixView(subMatrix, subView);
    }

    [TestMethod]
    public unsafe void F64MatrixView_LargePointerOffset()
    {
        const double testValue = 45;
        var largeMatrix = LargeMatrix();

        largeMatrix[largeMatrix.RowCount - 1, 0] = testValue;

        using var pinnedMatrix = largeMatrix.GetPinnedPointer();
        var matrixView = pinnedMatrix.View();

        var lastValue = *matrixView[largeMatrix.RowCount - 1];

        Assert.AreEqual(lastValue, testValue);
    }

    [TestMethod]
    public void F64MatrixView_ColumnLargePointerOffset()
    {
        const double testValue = 45;
        var largeMatrix = LargeMatrix();

        largeMatrix[largeMatrix.RowCount - 1, 0] = testValue;

        using var pinnedMatrix = largeMatrix.GetPinnedPointer();
        var columnView = pinnedMatrix.View().ColumnView(0);

        var lastValue = columnView[largeMatrix.RowCount - 1];

        Assert.AreEqual(lastValue, testValue);
    }

    /// <remarks>A matrix which needs a byte pointer offset larger than
    /// int.MaxValue to access all records in the backing array</remarks>
    static F64Matrix LargeMatrix()
    {
        return new F64Matrix(int.MaxValue / sizeof(double) + 2, 1);
    }

    static F64Matrix Matrix()
    {
        var features = new double[9] { 1, 2, 3,
                                    10, 20, 30,
                                    100, 200, 300,};

        return new F64Matrix(features, 3, 3);
    }

    static unsafe void AssertColumnView(double[] column, F64MatrixColumnView columnView)
    {
        for (var i = 0; i < column.Length; i++)
        {
            Assert.AreEqual(column[i], columnView[i]);
        }
    }

    unsafe void AssertMatrixView(IMatrix<double> matrix, F64MatrixView view)
    {
        for (var i = 0; i < matrix.RowCount; i++)
        {
            for (var j = 0; j < matrix.ColumnCount; j++)
            {
                Assert.AreEqual(matrix.At(i, j), view[i][j]);
            }
        }
    }
}
