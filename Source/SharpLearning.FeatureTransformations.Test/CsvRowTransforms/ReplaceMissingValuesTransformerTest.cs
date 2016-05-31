using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.CsvRowTransforms;
using SharpLearning.InputOutput.Csv;
using System.IO;

namespace SharpLearning.FeatureTransformations.Test.CsvRowTransforms
{
    /// <summary>
    /// Summary description for ReplaceMissingValuesTransformerTest
    /// </summary>
    [TestClass]
    public class ReplaceMissingValuesTransformerTest
    {
        [TestMethod]
        public void ReplaceMissingValuesTransformer_Transform()
        {
            var sut = new ReplaceMissingValuesTransformer("666", "", "NA", "na");

            var writer = new StringWriter();

            new CsvParser(() => new StringReader(Input))
            .EnumerateRows()
            .Transform(r => sut.Transform(r))
            .Write(() => writer);

            var actual = writer.ToString();

            Assert.AreEqual(Expected, actual);
        }

        string Expected =
@"Day
Monday
666
WednessDay
666
666
TuesDay";

        string Input =
@"Day
Monday

WednessDay
NA
na
TuesDay
";
    }
}
