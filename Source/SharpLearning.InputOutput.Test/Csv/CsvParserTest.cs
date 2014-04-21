using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.Collections.Generic;
using System.IO;
using SharpLearning.InputOutput.Test.Properties;
using System.Linq;

namespace SharpLearning.InputOutput.Test.Csv
{
    [TestClass]
    public class CsvParserTest
    {
        [TestMethod]
        public void CsvParser_EnumerateRows()
        {
            var sut = new CsvParser(() => new StringReader(Resources.AptitudeTestData));

            var actual = sut.EnumerateRows()
                            .Skip(10).Take(3)
                            .ToList();

            CollectionAssert.AreEqual(Expected(), actual);
        }

        [TestMethod]
        public void CsvParser_EnumerateRows_ColumNames()
        {
            var sut = new CsvParser(() => new StringReader(Resources.AptitudeTestData));

            var actual = sut.EnumerateRows("PreviousExperience_month", "Pass")
                            .Skip(10).Take(3)
                            .ToList();

            CollectionAssert.AreEqual(Expected_ColumnNames(), actual);
        }

        [TestMethod]
        public void CsvParser_EnumerateRows_Select_ColumNames()
        {
            var sut = new CsvParser(() => new StringReader(Resources.AptitudeTestData));

            var actual = sut.EnumerateRows(name => name == "Pass")
                            .Skip(10).Take(3)
                            .ToList();

            CollectionAssert.AreEqual(Expected_Select_ColumnNames(), actual);
        }

        public void DemilitedCsvParser_EnumerateRows_Quote_Inclosed_Columns()
        {
            var data = "\"c1\";\"c2\";\"c3\"" + Environment.NewLine +
                       "\"1\";\"2\";\"3\"" + Environment.NewLine +
                       "\"10\";\"20\";\"30\"" + Environment.NewLine;

            var sut = new CsvParser(() => new StringReader(data));

            var actual = sut.EnumerateRows()
                            .ToList();

            CollectionAssert.AreEqual(Expected_Quote_Inclosed_Columns(), actual);
        }

        List<CsvRow> Expected()
        {
            var columnNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, { "PreviousExperience_month", 1 }, { "Pass", 2 } };

            var expected = new List<CsvRow> { new CsvRow(new string[] { "5", "2", "1"}, columnNameToIndex),
                                                       new CsvRow(new string[] { "1", "12", "0"}, columnNameToIndex),
                                                       new CsvRow(new string[] { "3", "18", "0"}, columnNameToIndex) };

            return expected;
        }

        List<CsvRow> Expected_ColumnNames()
        {
            var columnNameToIndex = new Dictionary<string, int> { { "PreviousExperience_month", 0 }, { "Pass", 1 } };

            var expected = new List<CsvRow> { new CsvRow(new string[] { "2", "1"}, columnNameToIndex),
                                                       new CsvRow(new string[] { "12", "0"}, columnNameToIndex),
                                                       new CsvRow(new string[] { "18", "0"}, columnNameToIndex) };

            return expected;
        }

        List<CsvRow> Expected_Select_ColumnNames()
        {
            var columnNameToIndex = new Dictionary<string, int> { { "Pass", 0 } };

            var expected = new List<CsvRow> { new CsvRow(new string[] { "1" }, columnNameToIndex),
                                                       new CsvRow(new string[] { "0" }, columnNameToIndex),
                                                       new CsvRow(new string[] { "0" }, columnNameToIndex) };

            return expected;
        }

        List<CsvRow> Expected_Quote_Inclosed_Columns()
        {
            var columnNameToIndex = new Dictionary<string, int> { { "c1", 0 }, { "c2", 1 }, { "c3", 2 } };

            var expected = new List<CsvRow> { new CsvRow(new string[] { "1", "2", "3"}, columnNameToIndex),
                                                       new CsvRow(new string[] { "10", "20", "30"}, columnNameToIndex) };

            return expected;
        }
    }
}
