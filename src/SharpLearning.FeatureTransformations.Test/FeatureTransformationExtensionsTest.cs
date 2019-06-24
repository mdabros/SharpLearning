using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.FeatureTransformations.CsvRowTransforms;
using SharpLearning.FeatureTransformations.MatrixTransforms;
using SharpLearning.InputOutput.Csv;

namespace SharpLearning.FeatureTransformations.Test
{
    [TestClass]
    public class FeatureTransformationExtensionsTest
    {
        readonly string m_input =
@"Day;Sales;Open
Monday;123;Yes
TuesDay;NA;No
WednessDay;124;Yes
Monday;51;No
TuesDay;NA;Yes
";
        readonly string m_expected =
@"Day;Sales;Open;Day_Monday;Day_TuesDay;Day_WednessDay;Open_Yes;Open_No
Monday;123;Yes;1;0;0;1;0
TuesDay;666;No;0;1;0;0;1
WednessDay;124;Yes;0;0;1;1;0
Monday;51;No;1;0;0;0;1
TuesDay;666;Yes;0;1;0;1;0";

        [TestMethod]
        public void FeatureTransformationExtensions_MatrixTransform()
        {
            var meanZeroTransformer = new MeanZeroFeatureTransformer();
            var minMaxTransformer = new MinMaxTransformer(-1.0, 1.0);
            var matrix = new F64Matrix(new double[] { 123, 12,
                                                      41, 120,
                                                      124, 122 }, 3, 2);

            var actual = matrix.Transform(meanZeroTransformer.Transform)
                               .Transform(minMaxTransformer.Transform);

            var expected = new F64Matrix(new double[] { 0.97590361445783125, -1, -1, 0.96363636363636362, 1, 1 }, 3, 2);

            Assert.AreEqual(expected, actual);
        }


        [TestMethod]
        public void FeatureTransformationExtensions_RowTransform()
        {
            var replaceMissingTransformer = new ReplaceMissingValuesTransformer("666", "NA");
            var oneHotTransformer = new OneHotTransformer("Day", "Open");

            var writer = new StringWriter();

            new CsvParser(() => new StringReader(m_input))
            .EnumerateRows()
            .Transform(r => replaceMissingTransformer.Transform(r))
            .Transform(r => oneHotTransformer.Transform(r))
            .Write(() => writer);

            var actual = writer.ToString();

            Assert.AreEqual(m_expected, actual);
        }
    }
}
