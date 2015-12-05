using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.Containers.Test.Matrices
{
    [TestClass]
    public class StringMatrixTest
    {
        [TestMethod]
        public void StringMatrix_GetItemAt()
        {
            var sut = CreateFeatures();
            var item = sut.GetItemAt(1, 1);
            Assert.AreEqual("20", item);
        }

        [TestMethod]
        public void StringMatrix_GetItemAt_Indexer()
        {
            var sut = CreateFeatures();
            Assert.AreEqual("1", sut[0, 0]);
            Assert.AreEqual("100", sut[2, 0]);
            Assert.AreEqual("2", sut[0, 1]);
            Assert.AreEqual("200", sut[2, 1]);
            Assert.AreEqual("3", sut[0, 2]);
            Assert.AreEqual("300", sut[2, 2]);
        }


        [TestMethod]
        public void StringMatrix_SetItemAt()
        {
            var sut = CreateFeatures();
            var item = "123.0";
            sut.SetItemAt(1, 1, item);

            var value = sut.GetItemAt(1, 1);
            Assert.AreEqual(item, value);
        }

        [TestMethod]
        public void StringMatrix_SetItemAt_Indexer()
        {
            var sut = CreateFeatures();
            var item = "123.0";
            sut[1, 1]= item;

            var value = sut.GetItemAt(1, 1);
            Assert.AreEqual(item, value);
        }

        [TestMethod]
        public void StringMatrix_GetRow()
        {
            var sut = CreateFeatures();
            var row = sut.GetRow(1);
            CollectionAssert.AreEqual(GetExpectedRow(), row);
        }

        [TestMethod]
        public void StringMatrix_GetRow_Predefined()
        {
            var sut = CreateFeatures();
            var row = new string[sut.GetNumberOfColumns()];
            sut.GetRow(1, row);
            CollectionAssert.AreEqual(GetExpectedRow(), row);
        }

        [TestMethod]
        public void StringMatrix_GetColumn()
        {
            var sut = CreateFeatures();
            var col = sut.GetColumn(1);

            CollectionAssert.AreEqual(GetExpectedColumn(), col);
        }

        [TestMethod]
        public void StringMatrix_GetColumn_Predefined()
        {
            var sut = CreateFeatures();
            var col = new string[sut.GetNumberOfRows()];
            sut.GetColumn(1, col);

            CollectionAssert.AreEqual(GetExpectedColumn(), col);
        }

        [TestMethod]
        public void StringMatrix_GetRows()
        {
            var sut = CreateFeatures();
            var subMatrix = sut.GetRows(0, 2);
            var expected = GetExpectedRowSubMatrix();

            Assert.IsTrue(expected.Equals(subMatrix));
        }

        [TestMethod]
        public void StringMatrix_GetRows_Predefined()
        {
            var sut = CreateFeatures();
            var actual = new StringMatrix(2, 3);
            sut.GetRows(new int[] { 0, 2 }, actual);
            var expected = GetExpectedRowSubMatrix();

            Assert.IsTrue(expected.Equals(actual));
        }

        [TestMethod]
        public void StringMatrix_GetColumns()
        {
            var sut = CreateFeatures();
            var subMatrix = sut.GetColumns(0, 2);
            var expected = GetExpectedColSubMatrix();

            Assert.IsTrue(expected.Equals(subMatrix));
        }

        [TestMethod]
        public void StringMatrix_GetColumns_Predefined()
        {
            var sut = CreateFeatures();
            var actual = new StringMatrix(3, 2);
            sut.GetColumns(new int [] { 0, 2 }, actual);
            var expected = GetExpectedColSubMatrix();

            Assert.IsTrue(expected.Equals(actual));
        }

        string[] GetExpectedColumn()
        {
            return new string[] { "2", "20", "200" };
        }

        string[] GetExpectedRow()
        {
            return new string[] { "10", "20", "30" };
        }

        StringMatrix GetExpectedColSubMatrix()
        {
            var features = new string[] { "1", "3",
                                          "10", "30",
                                          "100", "300"};

            return new StringMatrix(features, 3, 2);
        }

        StringMatrix GetExpectedRowSubMatrix()
        {
            var features = new string[] { "1", "2", "3",
                                          "100", "200", "300"};

            return new StringMatrix(features, 2, 3);
        }

        StringMatrix CreateFeatures()
        {
            var features = new string[] { "1", "2", "3",
                                          "10", "20", "30",
                                          "100", "200", "300"};

            return new StringMatrix(features, 3, 3);
        }
    }
}
