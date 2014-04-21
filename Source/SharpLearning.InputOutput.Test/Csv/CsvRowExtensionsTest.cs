using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.Collections.Generic;
using SharpLearning.Containers.Matrices;
using System.Linq;
using SharpLearning.Containers;
using System.IO;

namespace SharpLearning.InputOutput.Test.Csv
{
    [TestClass]
    public class CsvRowExtensionsTest
    {
        static readonly string[] Data = new string[] { "1", "2", "3", "4" };
        static readonly Dictionary<string, int> ColumnNameToIndex = new Dictionary<string, int> { { "1", 0 }, { "2", 1 }, { "3", 2 }, { "4", 3 } };
        readonly F64Matrix ExpectedF64Matrix = new F64Matrix(Data.Select(value => FloatingPointConversion.ToF64(value)).ToArray(), 1, 4);
        readonly string ExpectedWrite = "1;2;3;4\r\n1;2;3;4";

        [TestMethod]
        public void CsvRowExtensions_ToF64Matrix()
        {  
            var sut = new List<CsvRow> { new CsvRow(Data, ColumnNameToIndex) };
            var actual = sut.ToF64Matrix();
            Assert.AreEqual(ExpectedF64Matrix, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_ToF64Vector()
        {
            var text = "one;two;three;four" + Environment.NewLine +
                       "1;2;3;4";

            var sut = new CsvParser(() => new StringReader(text));

            var actual = sut.EnumerateRows("one")
                            .ToF64Vector();

            CollectionAssert.AreEqual(new double[] { 1 }, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_Write()
        {
            var sut = new List<CsvRow> { new CsvRow(Data, ColumnNameToIndex) };

            var writer = new StringWriter();
            sut.Write(writer);

            var actual = writer.ToString();
            Assert.AreEqual(ExpectedWrite, actual);
        }
    }
}
