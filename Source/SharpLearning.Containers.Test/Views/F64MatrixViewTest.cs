using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;

namespace SharpLearning.Containers.Test.Views
{
    [TestClass]
    public class F64MatrixViewTest
    {
        [TestMethod]
        public unsafe void F64MatrixView_Indexing()
        {
            var matrix = Matrix();
            using (var pinnedMatrix = matrix.GetPinnedPointer())
            {
                var view = pinnedMatrix.View();
                AssertMatrixView(matrix, view);
            }
        }

        [TestMethod]
        public unsafe void F64MatrixView_ColumnView()
        {
            var matrix = Matrix();
            using (var pinnedMatrix = matrix.GetPinnedPointer())
            {
                var view = pinnedMatrix.View();
                for (int i = 0; i < matrix.ColumnCount; i++)
			    {
                    AssertColumnView(matrix.Column(i), view.ColumnView(i));
			    }                
            }
        }

        [TestMethod]
        public unsafe void F64MatrixView_SubView()
        {
            var matrix = Matrix();
            using (var pinnedMatrix = matrix.GetPinnedPointer())
            {
                var subView = pinnedMatrix.View().View(Interval2D.Create(Interval1D.Create(0, 2), Interval1D.Create(0, 3)));
                var subMatrix = matrix.Rows(new int[] { 0, 1 });
                AssertMatrixView(subMatrix, subView);
            }
        }

        F64Matrix Matrix()
        {
            var features = new double[9] { 1, 2, 3,
                                        10, 20, 30,
                                        100, 200, 300};

            return new F64Matrix(features, 3, 3);
        }

        unsafe void AssertColumnView(double[] column, F64MatrixColumnView columnView)
        {
            for (int i = 0; i < column.Length; i++)
            {
                Assert.AreEqual(column[i], columnView[i]);
            }
        }


        unsafe void AssertMatrixView(IMatrix<double> matrix, F64MatrixView view)
        {
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    Assert.AreEqual(matrix.At(i, j), view[i][j]);
                }
            }
        }
    }
}
