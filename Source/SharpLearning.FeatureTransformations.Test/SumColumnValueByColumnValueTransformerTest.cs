using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test
{
    /// <summary>
    /// Summary description for CountColumnValuesByColumnTransformerTest
    /// </summary>
    [TestClass]
    public class SumColumnValueByColumnValueTransformerTest
    {
        [TestMethod]
        public void SumColumnValueByColumnValueTransformer_Transform()
        {
            var sut = new SumColumnValueByColumnValueTransformer("Id", "666");

            var writer = new StringWriter();

            new CsvParser(() => new StringReader(Input))
            .EnumerateRows()
            .Transform(r => sut.Transform(r))
            .Write(() => writer);

            var actual = writer.ToString();

            Assert.AreEqual(Expected, actual);
        }

        string Expected =
@"Id;Day;Id_666_Counts
1;Monday;1
2;666;2
1;WednessDay;1
2;666;2
1;666;1
2;TuesDay;2";

        string Input =
@"Id;Day
1;Monday
2;666
1;WednessDay
2;666
1;666
2;TuesDay";
    }
}
