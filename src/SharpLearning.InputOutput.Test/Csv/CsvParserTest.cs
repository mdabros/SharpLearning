using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.InputOutput.Test.Csv;

[TestClass]
public class CsvParserTest
{
    [TestMethod]
    public void CsvParser_EnumerateRows()
    {
        var sut = new CsvParser(() => new StringReader(DataSetUtilities.AptitudeData));

        var actual = sut.EnumerateRows()
                        .Skip(10).Take(3)
                        .ToList();

        CollectionAssert.AreEqual(Expected(), actual);
    }

    [TestMethod]
    public void CsvParser_EnumerateRows_ColumNames()
    {
        var sut = new CsvParser(() => new StringReader(DataSetUtilities.AptitudeData));

        var actual = sut.EnumerateRows("PreviousExperience_month", "Pass")
                        .Skip(10).Take(3)
                        .ToList();

        CollectionAssert.AreEqual(Expected_ColumnNames(), actual);
    }

    [TestMethod]
    public void CsvParser_EnumerateRows_Select_ColumNames()
    {
        var sut = new CsvParser(() => new StringReader(DataSetUtilities.AptitudeData));

        var actual = sut.EnumerateRows(name => name == "Pass")
                        .Skip(10).Take(3)
                        .ToList();

        CollectionAssert.AreEqual(Expected_Select_ColumnNames(), actual);
    }

    [TestMethod]
    public void CsvParser_EnumerateRows_Quote_Inclosed_Columns()
    {
        var data = "\"c1\";\"c2\";\"c3\"" + Environment.NewLine +
                   "\"1\";\"2\";\"3\"" + Environment.NewLine +
                   "\"10\";\"20\";\"30\"" + Environment.NewLine;

        var sut = new CsvParser(() => new StringReader(data));

        var actual = sut.EnumerateRows()
                        .ToList();

        CollectionAssert.AreEqual(Expected_Quote_Inclosed_Columns(), actual);
    }

    [TestMethod]
    public void CsvParser_EnumerateRows_Quote_Inclosed_Columns_With_Separator_In_Text()
    {
        var data = "\"c1\";\"c2\";\"c3\"" + Environment.NewLine +
                   "\"1\";\"the following dates;1. jan, 1. april\";\"3\"" + Environment.NewLine +
                   "\"10\";\"20\";\"30\"" + Environment.NewLine;

        var sut = new CsvParser(() => new StringReader(data), ';', true);

        var actual = sut.EnumerateRows()
                        .ToList();

        CollectionAssert.AreEqual(Expected_Quote_Inclosed_Columns_Separator_In_Text(), actual);
    }

    [TestMethod]
    public void CsvParser_NoHeader_EnumerateRows()
    {
        var data = @"1;15;0
1;12;0
4;6;0";

        var sut = new CsvParser(() => new StringReader(data), ';', false, false);

        var actual = sut.EnumerateRows()
            .ToList();

        CollectionAssert.AreEqual(Expected_NoHeader(), actual);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void CsvParser_NoHeader_EnumerateRows_Func_Throw()
    {
        var sut = new CsvParser(() => new StringReader(string.Empty), ';', false, false);

        var actual = sut.EnumerateRows(p => p.Contains("test"));

        CollectionAssert.AreEqual(Expected_NoHeader(), actual.ToList());
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void CsvParser_NoHeader_EnumerateRows_Value_Throw()
    {
        var sut = new CsvParser(() => new StringReader(string.Empty), ';', false, false);

        var actual = sut.EnumerateRows("test");

        CollectionAssert.AreEqual(Expected_NoHeader(), actual.ToList());
    }

    static List<CsvRow> Expected_NoHeader()
    {
        var columnNameToIndex = new Dictionary<string, int> { { "0", 0 }, { "1", 1 }, { "2", 2 } };

        var expected = new List<CsvRow>
        {
            new(columnNameToIndex, new string[] { "1", "15", "0"}),
            new(columnNameToIndex, new string[] { "1", "12", "0"}),
            new(columnNameToIndex, new string[] { "4", "6", "0"})
        };

        return expected;
    }

    static List<CsvRow> Expected()
    {
        var columnNameToIndex = new Dictionary<string, int>
        {
            { "AptitudeTestScore", 0 },
            { "PreviousExperience_month", 1 },
            { "Pass", 2 }
        };

        var expected = new List<CsvRow>
        {
            new(columnNameToIndex, new string[] { "5", "2", "1"}),
            new(columnNameToIndex, new string[] { "1", "12", "0"}),
            new(columnNameToIndex, new string[] { "3", "18", "0"})
        };

        return expected;
    }

    static List<CsvRow> Expected_ColumnNames()
    {
        var columnNameToIndex = new Dictionary<string, int>
        {
            { "PreviousExperience_month", 0 },
            { "Pass", 1 }
        };

        var expected = new List<CsvRow>
        {
            new(columnNameToIndex, new string[] { "2", "1"}),
            new(columnNameToIndex, new string[] { "12", "0"}),
            new(columnNameToIndex, new string[] { "18", "0"})
        };

        return expected;
    }

    static List<CsvRow> Expected_Select_ColumnNames()
    {
        var columnNameToIndex = new Dictionary<string, int> { { "Pass", 0 } };

        var expected = new List<CsvRow>
        {
            new(columnNameToIndex, new string[] { "1" }),
            new(columnNameToIndex, new string[] { "0" }),
            new(columnNameToIndex, new string[] { "0" })
        };

        return expected;
    }

    static List<CsvRow> Expected_Quote_Inclosed_Columns()
    {
        var columnNameToIndex = new Dictionary<string, int> { { "c1", 0 }, { "c2", 1 }, { "c3", 2 } };

        var expected = new List<CsvRow>
        {
            new(columnNameToIndex, new string[] { "1", "2", "3"}),
            new(columnNameToIndex, new string[] { "10", "20", "30"})
        };

        return expected;
    }

    static List<CsvRow> Expected_Quote_Inclosed_Columns_Separator_In_Text()
    {
        var columnNameToIndex = new Dictionary<string, int> { { "c1", 0 }, { "c2", 1 }, { "c3", 2 } };

        var expected = new List<CsvRow>
        {
            new(columnNameToIndex, new string[] { "1", "the following dates;1. jan, 1. april", "3"}),
            new(columnNameToIndex, new string[] { "10", "20", "30"})
        };

        return expected;
    }
}
