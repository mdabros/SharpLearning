using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.InputOutput.Test.Csv
{
    /// <summary>
    /// Summary description for CsvRowTest
    /// </summary>
    [TestClass]
    public class CsvRowTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void CsvRow_Constructor_data_columnNames_does_not_match()
        {
            var row = new CsvRow(new string[] { "a", "b", "c" }, new Dictionary<string, int> { { "F1", 0 }, { "F2", 0 } });            
        }

        [TestMethod]
        public void CsvRow_Equal()
        {
            var row = new CsvRow(new string[] { "a", "b", "c" }, new Dictionary<string, int> { { "F1", 0 }, { "F2", 0 }, { "F3", 0 }, });
            var equal = new CsvRow(new string[] { "a", "b", "c" }, new Dictionary<string, int> { { "F1", 0 }, { "F2", 0 }, { "F3", 0 }, });
            var notEqual = new CsvRow(new string[] { "123", "b", "c" }, new Dictionary<string, int> { { "F1", 0 }, { "F2", 0 }, { "F3", 0 }, });

            Assert.AreEqual(equal, row);
            Assert.AreNotEqual(notEqual, row);
        }
    }
}
