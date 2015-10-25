using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test
{
    [TestClass]
    public class MapCategoricalFeaturesTransformerTest
    {
        [TestMethod]
        public void MapCategoricalFeaturesTransformer_Transform()
        {
            var sut = new MapCategoricalFeaturesTransformer();

            var writer = new StringWriter();

            new CsvParser(() => new StringReader(Input))
            .EnumerateRows()
            .Transform(r => sut.Transform(r, "Day"))
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
