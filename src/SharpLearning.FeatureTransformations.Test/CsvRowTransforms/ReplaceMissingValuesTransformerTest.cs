using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.CsvRowTransforms;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test.CsvRowTransforms;

/// <summary>
/// Summary description for ReplaceMissingValuesTransformerTest
/// </summary>
[TestClass]
public class ReplaceMissingValuesTransformerTest
{
    readonly string m_expected =
@"Day
Monday
666
WednessDay
666
666
TuesDay";

    readonly string m_input =
@"Day
Monday

WednessDay
NA
na
TuesDay
";

    [TestMethod]
    public void ReplaceMissingValuesTransformer_Transform()
    {
        var sut = new ReplaceMissingValuesTransformer("666", "", "NA", "na");

        var writer = new StringWriter();

        new CsvParser(() => new StringReader(m_input))
        .EnumerateRows()
        .Transform(r => sut.Transform(r))
        .Write(() => writer);

        var actual = writer.ToString();

        Assert.AreEqual(m_expected, actual);
    }
}
