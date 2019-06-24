using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.InputOutput.Test.Csv
{
    [TestClass]
    public class CsvRowExtensionsTest
    {
        static readonly string[] m_data = new string[] { "1", "2", "3", "4" };
        static readonly Dictionary<string, int> m_columnNameToIndex = new Dictionary<string, int> { { "1", 0 }, { "2", 1 }, { "3", 2 }, { "4", 3 } };
        readonly F64Matrix m_expectedF64Matrix = new F64Matrix(m_data.Select(value => CsvRowExtensions.DefaultF64Converter(value)).ToArray(), 1, 4);
        readonly StringMatrix m_expectedStringMatrix = new StringMatrix(m_data, 1, 4);

        readonly string m_expectedWrite = "1;2;3;4\r\n1;2;3;4";

        [TestMethod]
        public void CsvRowExtensions_GetValues()
        {
            var sut = new CsvRow(m_columnNameToIndex, m_data);
            var actual = sut.GetValues(new string[] {"1", "3"});
            var expected = new string[] { "1", "3" };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_SetValue()
        {
            var sut = new CsvRow(m_columnNameToIndex, m_data.ToArray());
            sut.SetValue("3", "33");
            
            var actual = sut.GetValue("3");
            Assert.AreEqual("33", actual);
        }

        [TestMethod]
        public void CsvRowExtensions_GetValue()
        {
            var sut = new CsvRow(m_columnNameToIndex, m_data);
            var actual = sut.GetValue("3");
            var expected = "3";
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_Keep()
        {
            var sut = new List<CsvRow> { new CsvRow(m_columnNameToIndex, m_data) };

            var actual = sut.Keep("1", "2").ToList().First();
            var expected = new CsvRow(new Dictionary<string, int> { { "1", 0 }, { "2", 1 } }, new string[] { "1", "2" });

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_Remove()
        {
            var sut = new List<CsvRow> { new CsvRow(m_columnNameToIndex, m_data) };

            var actual = sut.Remove("3").ToList().First();
            var expected = new CsvRow(new Dictionary<string, int> { { "1", 0 }, { "2", 1 }, { "4", 2 } }, new string[] { "1", "2", "4" });

            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_ToF64Matrix()
        {
            var sut = new List<CsvRow> { new CsvRow(m_columnNameToIndex, m_data) };
            var actual = sut.ToF64Matrix();
            Assert.AreEqual(m_expectedF64Matrix, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_ToStringMatrix()
        {
            var sut = new List<CsvRow> { new CsvRow(m_columnNameToIndex, m_data) };
            var actual = sut.ToStringMatrix();
            Assert.AreEqual(m_expectedStringMatrix, actual);
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
        public void CsvRowExtensions_ToStringVector()
        {
            var text = "one;two;three;four" + Environment.NewLine +
                       "1;2;3;4";

            var sut = new CsvParser(() => new StringReader(text));

            var actual = sut.EnumerateRows("one")
                            .ToStringVector();

            CollectionAssert.AreEqual(new string[] { "1" }, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_Write()
        {
            var sut = new List<CsvRow> { new CsvRow(m_columnNameToIndex, m_data) };

            var writer = new StringWriter();
            sut.Write(() => writer);

            var actual = writer.ToString();
            Assert.AreEqual(m_expectedWrite, actual);
        }

        [TestMethod]
        public void CsvRowExtensions_KeyCombine_KeepRepeatedColumns()
        {
            var keyName = "Date";

            var parser1 = new CsvParser(() => new StringReader(DataSetUtilities.TimeData1));
            var parser2 = new CsvParser(() => new StringReader(DataSetUtilities.TimeData2));

            var rows = parser1.EnumerateRows()
                              .KeyCombine(parser2.EnumerateRows(), (r1, r2) => r1.GetValue(keyName) == r2.GetValue(keyName), false);

            var writer = new StringWriter();
            rows.Write(() => writer);
            var actual = writer.ToString();
            var expected = "Date;Open;High;Low;Close;Volume;Adj Close;Date_1;Open_1;High_1;Low_1;Close_1;Volume_1;Adj Close_1\r\n2014-04-29;38.01;39.68;36.80;38.00;294200;38.00;2014-04-29;22.05;22.44;21.72;21.78;81900;21.78\r\n2014-04-28;38.26;39.36;37.30;37.83;361900;37.83;2014-04-28;21.79;22.00;21.46;21.90;71100;21.90\r\n2014-04-25;38.33;39.04;37.88;38.00;342900;38.00;2014-04-25;22.10;22.48;21.67;21.78;77500;21.78\r\n2014-04-24;39.33;39.59;37.91;38.82;362200;38.82;2014-04-24;22.61;22.70;22.20;22.23;48700;22.23\r\n2014-04-23;38.98;39.58;38.50;38.88;245800;38.88;2014-04-23;22.26;22.95;22.16;22.60;99400;22.60\r\n2014-04-22;38.43;39.79;38.31;38.99;358000;38.99;2014-04-22;22.19;22.70;22.13;22.48;69200;22.48\r\n2014-04-21;38.05;38.74;37.77;38.41;316800;38.41;2014-04-21;22.28;22.54;22.05;22.24;31100;22.24\r\n2014-04-17;37.25;38.24;36.92;38.05;233700;38.05;2014-04-17;22.30;22.40;22.15;22.26;47400;22.26\r\n2014-04-16;36.37;37.27;36.17;37.26;144800;37.26;2014-04-16;22.59;22.74;22.09;22.35;46600;22.35\r\n2014-04-15;36.08;36.74;35.09;36.05;223100;36.05;2014-04-15;22.46;22.74;21.95;22.35;40800;22.35\r\n2014-04-14;36.55;36.90;35.33;36.02;296100;36.02;2014-04-14;22.65;22.82;22.16;22.45;84600;22.45\r\n2014-04-11;36.26;37.09;36.08;36.13;282700;36.13;2014-04-11;22.31;22.69;22.28;22.43;66600;22.43\r\n2014-04-10;37.06;37.16;36.13;36.46;309800;36.46;2014-04-10;23.11;23.25;22.39;22.56;88800;22.56\r\n2014-04-09;36.08;37.26;35.66;37.13;209400;37.13;2014-04-09;23.15;23.30;22.95;23.18;58600;23.18\r\n2014-04-08;35.50;36.16;35.28;35.85;215700;35.85;2014-04-08;23.04;23.68;23.00;23.11;56200;23.11\r\n2014-04-07;36.49;37.30;35.27;35.48;312400;35.48;2014-04-07;23.41;23.73;23.01;23.09;61500;23.09\r\n2014-04-04;38.39;38.90;36.60;36.93;306500;36.93;2014-04-04;24.00;24.05;23.37;23.44;188500;23.44\r\n2014-04-03;38.62;39.78;37.90;38.14;269800;38.14;2014-04-03;23.97;23.97;23.77;23.90;43600;23.90\r\n2014-04-02;38.66;38.84;38.04;38.56;398200;38.56;2014-04-02;23.70;23.92;23.51;23.88;74700;23.88\r\n2014-04-01;37.21;38.65;36.58;38.49;410900;38.49;2014-04-01;23.34;23.87;23.13;23.75;146100;23.75";
            Assert.AreEqual(expected, actual);


            var actualColumnNameToIndex = rows.First().ColumnNameToIndex;
            var expectedColumnNameToIndex = new Dictionary<string, int> { {"Date", 0}, {"Open", 1}, {"High", 2}, {"Low", 3}, {"Close", 4}, {"Volume", 5}, {"Adj Close", 6},
                                                                          {"Date_1", 7}, {"Open_1", 8}, {"High_1", 9}, {"Low_1", 10}, {"Close_1", 11}, {"Volume_1", 12}, {"Adj Close_1", 13}};

            CollectionAssert.AreEqual(expectedColumnNameToIndex, actualColumnNameToIndex);
        }

        [TestMethod]
        public void CsvRowExtensions_KeyCombine()
        {
            var keyName = "Date";

            var parser1 = new CsvParser(() => new StringReader(DataSetUtilities.TimeData1));
            var parser2 = new CsvParser(() => new StringReader(DataSetUtilities.TimeData21));

            var rows = parser1.EnumerateRows()
                              .KeyCombine(parser2.EnumerateRows(), (r1, r2) => r1.GetValue(keyName) == r2.GetValue(keyName));

            var writer = new StringWriter();
            rows.Write(() => writer);
            var actual = writer.ToString();
            var expected = "Date;Open;High;Low;Close;Volume;Adj Close;OpenOther;CloseOther\r\n2014-04-29;38.01;39.68;36.80;38.00;294200;38.00;22.05;21.78\r\n2014-04-28;38.26;39.36;37.30;37.83;361900;37.83;21.79;21.90\r\n2014-04-25;38.33;39.04;37.88;38.00;342900;38.00;22.10;21.78\r\n2014-04-24;39.33;39.59;37.91;38.82;362200;38.82;22.61;22.23\r\n2014-04-23;38.98;39.58;38.50;38.88;245800;38.88;22.26;22.60\r\n2014-04-22;38.43;39.79;38.31;38.99;358000;38.99;22.19;22.48\r\n2014-04-21;38.05;38.74;37.77;38.41;316800;38.41;22.28;22.24\r\n2014-04-17;37.25;38.24;36.92;38.05;233700;38.05;22.30;22.26\r\n2014-04-16;36.37;37.27;36.17;37.26;144800;37.26;22.59;22.35\r\n2014-04-15;36.08;36.74;35.09;36.05;223100;36.05;22.46;22.35\r\n2014-04-14;36.55;36.90;35.33;36.02;296100;36.02;22.65;22.45\r\n2014-04-11;36.26;37.09;36.08;36.13;282700;36.13;22.31;22.43\r\n2014-04-10;37.06;37.16;36.13;36.46;309800;36.46;23.11;22.56\r\n2014-04-09;36.08;37.26;35.66;37.13;209400;37.13;23.15;23.18\r\n2014-04-08;35.50;36.16;35.28;35.85;215700;35.85;23.04;23.11\r\n2014-04-07;36.49;37.30;35.27;35.48;312400;35.48;23.41;23.09\r\n2014-04-04;38.39;38.90;36.60;36.93;306500;36.93;24.00;23.44\r\n2014-04-03;38.62;39.78;37.90;38.14;269800;38.14;23.97;23.90\r\n2014-04-02;38.66;38.84;38.04;38.56;398200;38.56;23.70;23.88\r\n2014-04-01;37.21;38.65;36.58;38.49;410900;38.49;23.34;23.75";
            Assert.AreEqual(expected, actual);


            var actualColumnNameToIndex = rows.First().ColumnNameToIndex;
            var expectedColumnNameToIndex = new Dictionary<string, int> { {"Date", 0}, {"Open", 1}, {"High", 2}, {"Low", 3}, {"Close", 4}, {"Volume", 5}, {"Adj Close", 6},
                                                                          {"OpenOther", 7}, {"CloseOther", 8} };

            CollectionAssert.AreEqual(expectedColumnNameToIndex, actualColumnNameToIndex);
        }

        [TestMethod]
        public void CsvRowExtensions_KeyCombine_KeepRepeatedColumns_Dict()
        {
            var keyName = "Date";

            var parser1 = new CsvParser(() => new StringReader(DataSetUtilities.TimeData1));
            var parser2 = new CsvParser(() => new StringReader(DataSetUtilities.TimeData2));

            var rows = parser1.EnumerateRows()
                              .KeyCombine(parser2.EnumerateRows(), keyName, keyName, false);

            var writer = new StringWriter();
            rows.Write(() => writer);
            var actual = writer.ToString();
            var expected = "Date;Open;High;Low;Close;Volume;Adj Close;Date_1;Open_1;High_1;Low_1;Close_1;Volume_1;Adj Close_1\r\n2014-04-29;38.01;39.68;36.80;38.00;294200;38.00;2014-04-29;22.05;22.44;21.72;21.78;81900;21.78\r\n2014-04-28;38.26;39.36;37.30;37.83;361900;37.83;2014-04-28;21.79;22.00;21.46;21.90;71100;21.90\r\n2014-04-25;38.33;39.04;37.88;38.00;342900;38.00;2014-04-25;22.10;22.48;21.67;21.78;77500;21.78\r\n2014-04-24;39.33;39.59;37.91;38.82;362200;38.82;2014-04-24;22.61;22.70;22.20;22.23;48700;22.23\r\n2014-04-23;38.98;39.58;38.50;38.88;245800;38.88;2014-04-23;22.26;22.95;22.16;22.60;99400;22.60\r\n2014-04-22;38.43;39.79;38.31;38.99;358000;38.99;2014-04-22;22.19;22.70;22.13;22.48;69200;22.48\r\n2014-04-21;38.05;38.74;37.77;38.41;316800;38.41;2014-04-21;22.28;22.54;22.05;22.24;31100;22.24\r\n2014-04-17;37.25;38.24;36.92;38.05;233700;38.05;2014-04-17;22.30;22.40;22.15;22.26;47400;22.26\r\n2014-04-16;36.37;37.27;36.17;37.26;144800;37.26;2014-04-16;22.59;22.74;22.09;22.35;46600;22.35\r\n2014-04-15;36.08;36.74;35.09;36.05;223100;36.05;2014-04-15;22.46;22.74;21.95;22.35;40800;22.35\r\n2014-04-14;36.55;36.90;35.33;36.02;296100;36.02;2014-04-14;22.65;22.82;22.16;22.45;84600;22.45\r\n2014-04-11;36.26;37.09;36.08;36.13;282700;36.13;2014-04-11;22.31;22.69;22.28;22.43;66600;22.43\r\n2014-04-10;37.06;37.16;36.13;36.46;309800;36.46;2014-04-10;23.11;23.25;22.39;22.56;88800;22.56\r\n2014-04-09;36.08;37.26;35.66;37.13;209400;37.13;2014-04-09;23.15;23.30;22.95;23.18;58600;23.18\r\n2014-04-08;35.50;36.16;35.28;35.85;215700;35.85;2014-04-08;23.04;23.68;23.00;23.11;56200;23.11\r\n2014-04-07;36.49;37.30;35.27;35.48;312400;35.48;2014-04-07;23.41;23.73;23.01;23.09;61500;23.09\r\n2014-04-04;38.39;38.90;36.60;36.93;306500;36.93;2014-04-04;24.00;24.05;23.37;23.44;188500;23.44\r\n2014-04-03;38.62;39.78;37.90;38.14;269800;38.14;2014-04-03;23.97;23.97;23.77;23.90;43600;23.90\r\n2014-04-02;38.66;38.84;38.04;38.56;398200;38.56;2014-04-02;23.70;23.92;23.51;23.88;74700;23.88\r\n2014-04-01;37.21;38.65;36.58;38.49;410900;38.49;2014-04-01;23.34;23.87;23.13;23.75;146100;23.75";
            Assert.AreEqual(expected, actual);


            var actualColumnNameToIndex = rows.First().ColumnNameToIndex;
            var expectedColumnNameToIndex = new Dictionary<string, int> { {"Date", 0}, {"Open", 1}, {"High", 2}, {"Low", 3}, {"Close", 4}, {"Volume", 5}, {"Adj Close", 6},
                                                                          {"Date_1", 7}, {"Open_1", 8}, {"High_1", 9}, {"Low_1", 10}, {"Close_1", 11}, {"Volume_1", 12}, {"Adj Close_1", 13}};

            CollectionAssert.AreEqual(expectedColumnNameToIndex, actualColumnNameToIndex);
        }

        [TestMethod]
        public void CsvRowExtensions_KeyCombine_Dict()
        {
            var keyName = "Date";

            var parser1 = new CsvParser(() => new StringReader(DataSetUtilities.TimeData1));
            var parser2 = new CsvParser(() => new StringReader(DataSetUtilities.TimeData21));

            var rows = parser1.EnumerateRows()
                              .KeyCombine(parser2.EnumerateRows(), keyName, keyName);

            var writer = new StringWriter();
            rows.Write(() => writer);
            var actual = writer.ToString();
            var expected = "Date;Open;High;Low;Close;Volume;Adj Close;OpenOther;CloseOther\r\n2014-04-29;38.01;39.68;36.80;38.00;294200;38.00;22.05;21.78\r\n2014-04-28;38.26;39.36;37.30;37.83;361900;37.83;21.79;21.90\r\n2014-04-25;38.33;39.04;37.88;38.00;342900;38.00;22.10;21.78\r\n2014-04-24;39.33;39.59;37.91;38.82;362200;38.82;22.61;22.23\r\n2014-04-23;38.98;39.58;38.50;38.88;245800;38.88;22.26;22.60\r\n2014-04-22;38.43;39.79;38.31;38.99;358000;38.99;22.19;22.48\r\n2014-04-21;38.05;38.74;37.77;38.41;316800;38.41;22.28;22.24\r\n2014-04-17;37.25;38.24;36.92;38.05;233700;38.05;22.30;22.26\r\n2014-04-16;36.37;37.27;36.17;37.26;144800;37.26;22.59;22.35\r\n2014-04-15;36.08;36.74;35.09;36.05;223100;36.05;22.46;22.35\r\n2014-04-14;36.55;36.90;35.33;36.02;296100;36.02;22.65;22.45\r\n2014-04-11;36.26;37.09;36.08;36.13;282700;36.13;22.31;22.43\r\n2014-04-10;37.06;37.16;36.13;36.46;309800;36.46;23.11;22.56\r\n2014-04-09;36.08;37.26;35.66;37.13;209400;37.13;23.15;23.18\r\n2014-04-08;35.50;36.16;35.28;35.85;215700;35.85;23.04;23.11\r\n2014-04-07;36.49;37.30;35.27;35.48;312400;35.48;23.41;23.09\r\n2014-04-04;38.39;38.90;36.60;36.93;306500;36.93;24.00;23.44\r\n2014-04-03;38.62;39.78;37.90;38.14;269800;38.14;23.97;23.90\r\n2014-04-02;38.66;38.84;38.04;38.56;398200;38.56;23.70;23.88\r\n2014-04-01;37.21;38.65;36.58;38.49;410900;38.49;23.34;23.75";
            Assert.AreEqual(expected, actual);


            var actualColumnNameToIndex = rows.First().ColumnNameToIndex;
            var expectedColumnNameToIndex = new Dictionary<string, int> { {"Date", 0}, {"Open", 1}, {"High", 2}, {"Low", 3}, {"Close", 4}, {"Volume", 5}, {"Adj Close", 6},
                                                                          {"OpenOther", 7}, {"CloseOther", 8} };

            CollectionAssert.AreEqual(expectedColumnNameToIndex, actualColumnNameToIndex);
        }
    }
}
