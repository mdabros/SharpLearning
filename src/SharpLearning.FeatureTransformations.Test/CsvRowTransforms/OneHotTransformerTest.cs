using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.CsvRowTransforms;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test.CsvRowTransforms
{
    [TestClass]
    public class OneHotTransformerTest
    {
        readonly string m_expected =
@"Day;Sales;Open;Day_Monday;Day_TuesDay;Day_WednessDay;Open_Yes;Open_No
Monday;123;Yes;1;0;0;1;0
TuesDay;41;No;0;1;0;0;1
WednessDay;124;Yes;0;0;1;1;0
Monday;51;No;1;0;0;0;1
TuesDay;12;Yes;0;1;0;1;0";

        readonly string m_expectedSpaces =
@"Day;Sales;Open;Day_Monday;Day_TuesDay;Day_WednessDay;Open_Yes;Open_No
Monday;123;Yes;1;0;0;1;0
 TuesDay ;41;No;0;1;0;0;1
WednessDay;124;Yes;0;0;1;1;0
Monday;51;No;1;0;0;0;1
TuesDay;12;Yes;0;1;0;1;0";

        readonly string m_input =
@"Day;Sales;Open
Monday;123;Yes
TuesDay;41;No
WednessDay;124;Yes
Monday;51;No
TuesDay;12;Yes
";

        readonly string m_inputSpaces =
@"Day;Sales;Open
Monday;123;Yes
 TuesDay ;41;No
WednessDay;124;Yes
Monday;51;No
TuesDay;12;Yes
";

        [TestMethod]
        public void OneHotTransformer_Transform()
        {
            var sut = new OneHotTransformer("Day", "Open");

            var writer = new StringWriter();

            new CsvParser(() => new StringReader(m_input))
            .EnumerateRows()
            .Transform(r => sut.Transform(r))
            .Write(() => writer);

            var actual = writer.ToString();

            Assert.AreEqual(m_expected, actual);
        }

        [TestMethod]
        public void OneHotTransformer_Transform_Leading_Trainling_Spaces()
        {
            var sut = new OneHotTransformer("Day", "Open");

            var writer = new StringWriter();

            new CsvParser(() => new StringReader(m_inputSpaces))
            .EnumerateRows()
            .Transform(r => sut.Transform(r))
            .Write(() => writer);

            var actual = writer.ToString();

            Assert.AreEqual(m_expectedSpaces, actual);
        }
    }
}
