using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using System;

namespace SharpLearning.Containers.Test.Matrices
{
    [TestClass]
    public class F64MatrixTest
    {
        [TestMethod]
        public void F64Matrix_At()
        {
            var sut = CreateFeatures();
            var item = sut.At(1, 1);
            Assert.AreEqual(20, item);
        }

        [TestMethod]
        public void F64Matrix_At_Indexer()
        {
            var sut = CreateFeatures();
            Assert.AreEqual(1, sut[0, 0]);
            Assert.AreEqual(100, sut[2, 0]);
            Assert.AreEqual(2, sut[0, 1]);
            Assert.AreEqual(200, sut[2, 1]);
            Assert.AreEqual(3, sut[0, 2]);
            Assert.AreEqual(300, sut[2, 2]);
        }


        [TestMethod]
        public void F64Matrix_At_Set()
        {
            var sut = CreateFeatures();
            var item = 123.0;
            sut.At(1, 1, item);

            var value = sut.At(1, 1);
            Assert.AreEqual(item, value);
        }

        [TestMethod]
        public void F64Matrix_At_Set_Indexer()
        {
            var sut = CreateFeatures();
            var item = 123.0;
            sut[1, 1] = item;

            var value = sut.At(1, 1);
            Assert.AreEqual(item, value);
        }


        [TestMethod]
        public void F64Matrix_Row()
        {
            var sut = CreateFeatures();
            var row = sut.Row(1);
            CollectionAssert.AreEqual(GetExpectedRow(), row);
        }

        [TestMethod]
        public void F64Matrix_Row_Predefined()
        {
            var sut = CreateFeatures();
            var row = new double[sut.ColumnCount];
            sut.Row(1, row);
            CollectionAssert.AreEqual(GetExpectedRow(), row);
        }

        [TestMethod]
        public void F64Matrix_Column()
        {
            var sut = CreateFeatures();
            var col = sut.Column(1);

            CollectionAssert.AreEqual(GetExpectedColumn(), col);
        }

        [TestMethod]
        public void F64Matrix_Column_Predefined()
        {
            var sut = CreateFeatures();
            var col = new double[sut.RowCount];
            sut.Column(1, col);

            CollectionAssert.AreEqual(GetExpectedColumn(), col);
        }

        [TestMethod]
        public void F64Matrix_Rows()
        {
            var sut = CreateFeatures();
            var subMatrix = sut.Rows(0, 2);
            var expected = GetExpectedRowSubMatrix();

            Assert.IsTrue(expected.Equals(subMatrix));
        }

        [TestMethod]
        public void F64Matrix_Rows_Predefined()
        {
            var sut = CreateFeatures();
            var actual = new F64Matrix(2, 3);
            sut.Rows(new int[] { 0, 2 }, actual);
            var expected = GetExpectedRowSubMatrix();

            Assert.IsTrue(expected.Equals(actual));
        }

        [TestMethod]
        public void F64Matrix_Columns()
        {
            var sut = CreateFeatures();
            var subMatrix = sut.Columns(0, 2);
            var expected = GetExpectedColSubMatrix();

            Assert.IsTrue(expected.Equals(subMatrix));
        }

        [TestMethod]
        public void F64Matrix_Columns_predefined()
        {
            var sut = CreateFeatures();
            var actual = new F64Matrix(3, 2);
            sut.Columns(new int[] { 0, 2 }, actual);
            var expected = GetExpectedColSubMatrix();

            Assert.IsTrue(expected.Equals(actual));
        }

        [TestMethod]
        public void F64Matrix_Implicit_Conversion()
        {
            Func<F64Matrix, F64Matrix> converter = m => m;

            var actual = converter(new double[][] { new double[] { 0, 1 }, new double[] { 2, 3 } });

           Assert.AreEqual(0, actual.At(0,0));
           Assert.AreEqual(1, actual.At(0,1));
           Assert.AreEqual(2, actual.At(1,0));
           Assert.AreEqual(3, actual.At(1,1));
        }

        double[] GetExpectedColumn()
        {
            return new double[3] { 2, 20, 200 };
        }

        double[] GetExpectedRow()
        {
            return new double[3] { 10, 20, 30 };
        }

        F64Matrix GetExpectedColSubMatrix()
        {
            var features = new double[6] { 1, 3,
                                        10, 30,
                                        100, 300};

            return new F64Matrix(features, 3, 2);
        }

        F64Matrix GetExpectedRowSubMatrix()
        {
            var features = new double[6] { 1, 2, 3,
                                        100, 200, 300};

            return new F64Matrix(features, 2, 3);
        }

        F64Matrix CreateFeatures()
        {
            var features = new double[9] { 1, 2, 3,
                                        10, 20, 30,
                                        100, 200, 300};

            return new F64Matrix(features, 3, 3);
        }
    }
}
