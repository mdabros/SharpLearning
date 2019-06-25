using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Metrics.Test.Classification
{
    [TestClass]
    public class ClassificationMatrixStringConverterTest
    {
        [TestMethod]
        public void ClassificationMatrixStringConverter_Convert()
        {
            var confusionMatrix = new int[,] { { 10, 0 }, { 0, 10 } };
            var errorMatrix = new double[,] { { 1.0, 0.0 }, { 1.0, 0.0 } };
            var uniqueTargets = new List<double> { 1.0, 2.0 };

            var actual = ClassificationMatrixStringConverter.Convert(uniqueTargets, confusionMatrix, errorMatrix, 0.0);

            var expected = ";1;2;1;2\r\n1;10.000;0.000;100.000;0.000\r\n2;0.000;10.000;100.000;0.000\r\nError: 0.000\r\n";
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationMatrixStringConverter_Convert_TargetStringMapping()
        {
            var confusionMatrix = new int[,] { { 10, 0 }, { 0, 10 } };
            var errorMatrix = new double[,] { { 1.0, 0.0 }, { 1.0, 0.0 } };
            var uniqueTargets = new List<double> { 1.0, 2.0 };
            var uniqueStringTargets = new List<string> { "Positive", "Negative" };

            var actual = ClassificationMatrixStringConverter.Convert(uniqueStringTargets, confusionMatrix, errorMatrix, 0.0);

            var expected = ";Positive;Negative;Positive;Negative\r\nPositive;10.000;0.000;100.000;0.000\r\nNegative;0.000;10.000;100.000;0.000\r\nError: 0.000\r\n";
            Assert.AreEqual(expected, actual);
        }
    }
}
