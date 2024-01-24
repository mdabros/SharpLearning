using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.CsvRowTransforms;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test.CsvRowTransforms
{
    /// <summary>
    /// Summary description for DateTimeFeatureTransformerTest
    /// </summary>
    [TestClass]
    public class DateTimeFeatureTransformerTest
    {
        readonly string m_expected =
@"Date;Year;Month;WeekOfYear;DayOfMonth;DayOfWeek;HourOfDay;TotalDays;TotalHours
2015-01-17;2015;1;3;17;6;0;16452;394848
2015-02-21;2015;2;8;21;6;0;16487;395688
2015-03-13;2015;3;11;13;5;0;16507;396168
2015-05-12;2015;5;20;12;2;0;16567;397608
2015-04-4;2015;4;14;4;6;0;16529;396696
2015-03-12;2015;3;11;12;4;0;16506;396144
2015-02-14;2015;2;7;14;6;0;16480;395520
2015-01-16;2015;1;3;16;5;0;16451;394824";

        readonly string m_input =
@"Date
2015-01-17
2015-02-21
2015-03-13
2015-05-12
2015-04-4
2015-03-12
2015-02-14
2015-01-16
";

        [TestMethod]
        public void DateTimeFeatureTransformer_Transform()
        {
            var sut = new DateTimeFeatureTransformer("Date");

            var writer = new StringWriter();

            new CsvParser(() => new StringReader(m_input))
            .EnumerateRows()
            .Transform(r => sut.Transform(r))
            .Write(() => writer);

            var actual = writer.ToString();

            Assert.AreEqual(m_expected, actual);
        }
    }
}
