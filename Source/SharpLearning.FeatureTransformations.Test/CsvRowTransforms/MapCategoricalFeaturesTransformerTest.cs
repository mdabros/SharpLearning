using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.CsvRowTransforms;
using SharpLearning.InputOutput.Csv;
using System.IO;

namespace SharpLearning.FeatureTransformations.Test.CsvRowTransforms
{
    [TestClass]
    public class MapCategoricalFeaturesTransformerTest
    {
        [TestMethod]
        public void MapCategoricalFeaturesTransformer_Transform()
        {
            var sut = new MapCategoricalFeaturesTransformer("Day");

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
0
1
2
0
1";

        string Input =
@"Day
Monday
TuesDay
WednessDay
Monday
TuesDay
";
    }
}
