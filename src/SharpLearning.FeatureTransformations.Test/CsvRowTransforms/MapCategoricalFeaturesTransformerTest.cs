using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.CsvRowTransforms;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test.CsvRowTransforms;

[TestClass]
public class MapCategoricalFeaturesTransformerTest
{
    readonly string m_expected =
@"Day
0
1
2
0
1";
    readonly string m_input =
@"Day
Monday
TuesDay
WednessDay
Monday
TuesDay
";

    [TestMethod]
    public void MapCategoricalFeaturesTransformer_Transform()
    {
        var sut = new MapCategoricalFeaturesTransformer("Day");

        var writer = new StringWriter();

        new CsvParser(() => new StringReader(m_input))
        .EnumerateRows()
        .Transform(r => sut.Transform(r))
        .Write(() => writer);

        var actual = writer.ToString();

        Assert.AreEqual(m_expected, actual);
    }
}
