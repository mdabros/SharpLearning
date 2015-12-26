using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.Normalization;
using System.IO;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test
{
    [TestClass]
    public class MeanZeroFeatureTransformerTest
    {
        [TestMethod]
        public void FeatureNormalizationTransformer_Transform()
        {
            var sut = new MeanZeroFeatureTransformer();

            var writer = new StringWriter();

            new CsvParser(() => new StringReader(Input))
            .EnumerateRows()
            .Transform(r => sut.Transform(r, "Sales", "Counts"))
            .Write(() => writer);

            var actual = writer.ToString();

            Assert.AreEqual(Expected, actual);
        }

        string Expected =
@"Sales;Day;Counts
80.636363636363626;Monday;-48.545454545454547
-1.3636363636363669;Monday;59.454545454545453
81.636363636363626;Monday;61.454545454545453
8.6363636363636331;Monday;-39.545454545454547
-30.363636363636367;Monday;38.454545454545453
-30.363636363636367;Monday;-26.545454545454547
-28.363636363636367;Monday;-26.545454545454547
-37.363636363636367;Monday;-4.5454545454545467
-41.363636363636367;Monday;-37.545454545454547
-36.363636363636367;Monday;63.454545454545453
34.636363636363633;Monday;-39.545454545454547";

        string Input =
@"Sales;Day;Counts
123;Monday;12
41;Monday;120
124;Monday;122
51;Monday;21
12;Monday;99
12;Monday;34
14;Monday;34
5;Monday;56
1;Monday;23
6;Monday;124
77;Monday;21
";
    }
}
