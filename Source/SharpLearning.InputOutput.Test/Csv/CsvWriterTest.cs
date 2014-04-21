using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.InputOutput.Test.Properties;

namespace SharpLearning.InputOutput.Test.Csv
{
    /// <summary>
    /// Summary description for CsvWriterTest
    /// </summary>
    [TestClass]
    public class CsvWriterTest
    {
        readonly string Expected = "AptitudeTestScore;PreviousExperience_month;Pass\r\n5;6;0\r\n1;15;0\r\n1;12;0\r\n4;6;0\r\n1;15;1\r\n1;6;0\r\n4;16;1\r\n1;10;1\r\n3;12;0\r\n4;26;1\r\n5;2;1\r\n1;12;0\r\n3;18;0\r\n3;3;0\r\n1;24;1\r\n2;8;0\r\n1;9;0\r\n4;18;0\r\n4;22;1\r\n5;3;1\r\n4;12;0\r\n4;24;1\r\n2;18;1\r\n2;6;0\r\n1;8;0\r\n5;12;0";

        [TestMethod]
        public void CsvWriter_Write()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeTestData));
            var data = parser.EnumerateRows();

            var writer = new StringWriter();
            var sut = new CsvWriter(writer);
            sut.Write(data);

            var actual = writer.ToString();
            Assert.AreEqual(Expected, actual);
        }
    }
}
