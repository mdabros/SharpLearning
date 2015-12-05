using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Test.Matrices
{
    [TestClass]
    public class F64MatrixTest
    {
        [TestMethod]
        public void F64Matrix_GetItemAt()
        {
            var sut = CreateFeatures();
            var item = sut.GetItemAt(1, 1);
            Assert.AreEqual(20, item);
        }

        [TestMethod]
        public void F64Matrix_GetItemAt_Indexer()
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
        public void F64Matrix_SetItemAt()
        {
            var sut = CreateFeatures();
            var item = 123.0;
            sut.SetItemAt(1, 1, item);

            var value = sut.GetItemAt(1, 1);
            Assert.AreEqual(item, value);
        }

        [TestMethod]
        public void F64Matrix_SetItemAt_Indexer()
        {
            var sut = CreateFeatures();
            var item = 123.0;
            sut[1, 1] = item;

            var value = sut.GetItemAt(1, 1);
            Assert.AreEqual(item, value);
        }


        [TestMethod]
        public void F64Matrix_GetRow()
        {
            var sut = CreateFeatures();
            var row = sut.GetRow(1);
            CollectionAssert.AreEqual(GetExpectedRow(), row);
        }

        [TestMethod]
        public void F64Matrix_GetRow_Predefined()
        {
            var sut = CreateFeatures();
            var row = new double[sut.GetNumberOfColumns()];
            sut.GetRow(1, row);
            CollectionAssert.AreEqual(GetExpectedRow(), row);
        }

        [TestMethod]
        public void F64Matrix_GetColumn()
        {
            var sut = CreateFeatures();
            var col = sut.GetColumn(1);

            CollectionAssert.AreEqual(GetExpectedColumn(), col);
        }

        [TestMethod]
        public void F64Matrix_GetColumn_Predefined()
        {
            var sut = CreateFeatures();
            var col = new double[sut.GetNumberOfRows()];
            sut.GetColumn(1, col);

            CollectionAssert.AreEqual(GetExpectedColumn(), col);
        }

        [TestMethod]
        public void F64Matrix_GetRows()
        {
            var sut = CreateFeatures();
            var subMatrix = sut.GetRows(0, 2);
            var expected = GetExpectedRowSubMatrix();

            Assert.IsTrue(expected.Equals(subMatrix));
        }

        [TestMethod]
        public void F64Matrix_GetRows_Predefined()
        {
            var sut = CreateFeatures();
            var actual = new F64Matrix(2, 3);
            sut.GetRows(new int [] { 0, 2}, actual);
            var expected = GetExpectedRowSubMatrix();

            Assert.IsTrue(expected.Equals(actual));
        }

        [TestMethod]
        public void F64Matrix_GetColumns()
        {
            var sut = CreateFeatures();
            var subMatrix = sut.GetColumns(0, 2);
            var expected = GetExpectedColSubMatrix();

            Assert.IsTrue(expected.Equals(subMatrix));
        }

        [TestMethod]
        public void F64Matrix_GetColumns_predefined()
        {
            var sut = CreateFeatures();
            var actual = new F64Matrix(3, 2);
            sut.GetColumns(new int[] { 0, 2 }, actual);
            var expected = GetExpectedColSubMatrix();

            Assert.IsTrue(expected.Equals(actual));
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
