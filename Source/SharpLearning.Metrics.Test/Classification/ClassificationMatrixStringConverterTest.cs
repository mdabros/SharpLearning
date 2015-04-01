using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;

namespace SharpLearning.Metrics.Test.Classification
{
    [TestClass]
    public class ClassificationMatrixStringConverterTest
    {
        [TestMethod]
        public void ClassificationMatrixStringConverter_Convert()
        {
            var confusionMatrix = new int[][] { new int[] { 10, 0 }, new int[] { 0, 10 } };
            var errorMatrix = new double[][] { new double[] { 1.0, 0.0 }, new double[] { 1.0, 0.0 } };
            var uniqueTargets = new List<double> { 1.0, 2.0 };

            var sut = new ClassificationMatrixStringConverter<double>();
            var actual = sut.Convert(uniqueTargets, confusionMatrix, errorMatrix, 0.0);

            var expected = ";1;2;1;2\r\n1;10.000;0.000;100.000;0.000\r\n2;0.000;10.000;100.000;0.000\r\nError: 0.000\r\n";
            Assert.AreEqual(expected, actual);
        }
    }
}
