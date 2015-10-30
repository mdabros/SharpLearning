using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.FeatureTransformations.Normalization;
using System.IO;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test.Normalization
{
    [TestClass]
    public class FeatureNormalizationTransformerTest
    {
        [TestMethod]
        public void FeatureNormalizationTransformer_Transform()
        {
            var sut = new FeatureNormalizationTransformer(-1.0, 1.0);

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
0.9837398373983739;Monday;-1
-0.34959349593495936;Monday;0.9285714285714286
1;Monday;0.96428571428571419
-0.18699186991869921;Monday;-0.8392857142857143
-0.82113821138211385;Monday;0.5535714285714286
-0.82113821138211385;Monday;-0.60714285714285721
-0.78861788617886175;Monday;-0.60714285714285721
-0.93495934959349591;Monday;-0.2142857142857143
-1;Monday;-0.8035714285714286
-0.91869918699186992;Monday;1
0.2357723577235773;Monday;-0.8392857142857143";

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
