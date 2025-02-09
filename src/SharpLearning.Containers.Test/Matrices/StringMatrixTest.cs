using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;

namespace SharpLearning.Containers.Test.Matrices;

[TestClass]
public class StringMatrixTest
{
    [TestMethod]
    public void StringMatrix_At()
    {
        var sut = CreateFeatures();
        var item = sut.At(1, 1);
        Assert.AreEqual("20", item);
    }

    [TestMethod]
    public void StringMatrix_At_Indexer()
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
    public void StringMatrix_At_Set()
    {
        var sut = CreateFeatures();
        var item = "123.0";
        sut.At(1, 1, item);

        var value = sut.At(1, 1);
        Assert.AreEqual(item, value);
    }

    [TestMethod]
    public void StringMatrix_At_Set_Indexer()
    {
        var sut = CreateFeatures();
        var item = "123.0";
        sut[1, 1] = item;

        var value = sut.At(1, 1);
        Assert.AreEqual(item, value);
    }

    [TestMethod]
    public void StringMatrix_Row()
    {
        var sut = CreateFeatures();
        var row = sut.Row(1);
        CollectionAssert.AreEqual(GetExpectedRow(), row);
    }

    [TestMethod]
    public void StringMatrix_Row_Predefined()
    {
        var sut = CreateFeatures();
        var row = new string[sut.ColumnCount];
        sut.Row(1, row);
        CollectionAssert.AreEqual(GetExpectedRow(), row);
    }

    [TestMethod]
    public void StringMatrix_Column()
    {
        var sut = CreateFeatures();
        var col = sut.Column(1);

        CollectionAssert.AreEqual(GetExpectedColumn(), col);
    }

    [TestMethod]
    public void StringMatrix_Column_Predefined()
    {
        var sut = CreateFeatures();
        var col = new string[sut.RowCount];
        sut.Column(1, col);

        CollectionAssert.AreEqual(GetExpectedColumn(), col);
    }

    [TestMethod]
    public void StringMatrix_Rows()
    {
        var sut = CreateFeatures();
        var subMatrix = sut.Rows(0, 2);
        var expected = GetExpectedRowSubMatrix();

        Assert.IsTrue(expected.Equals(subMatrix));
    }

    [TestMethod]
    public void StringMatrix_Rows_Predefined()
    {
        var sut = CreateFeatures();
        var actual = new StringMatrix(2, 3);
        sut.Rows([0, 2], actual);
        var expected = GetExpectedRowSubMatrix();

        Assert.IsTrue(expected.Equals(actual));
    }

    [TestMethod]
    public void StringMatrix_Columns()
    {
        var sut = CreateFeatures();
        var subMatrix = sut.Columns(0, 2);
        var expected = GetExpectedColSubMatrix();

        Assert.IsTrue(expected.Equals(subMatrix));
    }

    [TestMethod]
    public void StringMatrix_Columns_Predefined()
    {
        var sut = CreateFeatures();
        var actual = new StringMatrix(3, 2);
        sut.Columns([0, 2], actual);
        var expected = GetExpectedColSubMatrix();

        Assert.IsTrue(expected.Equals(actual));
    }

    static string[] GetExpectedColumn()
    {
        return ["2", "20", "200"];
    }

    static string[] GetExpectedRow()
    {
        return ["10", "20", "30"];
    }

    static StringMatrix GetExpectedColSubMatrix()
    {
        var features = new string[] { "1", "3",
                                      "10", "30",
                                      "100", "300"};

        return new StringMatrix(features, 3, 2);
    }

    static StringMatrix GetExpectedRowSubMatrix()
    {
        var features = new string[] { "1", "2", "3",
                                      "100", "200", "300"};

        return new StringMatrix(features, 2, 3);
    }

    static StringMatrix CreateFeatures()
    {
        var features = new string[] { "1", "2", "3",
                                      "10", "20", "30",
                                      "100", "200", "300"};

        return new StringMatrix(features, 3, 3);
    }
}
