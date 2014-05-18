using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Metrics.Test.Classification
{
    [TestClass]
    public class TotalErrorClassificationMetricTest
    {
        [TestMethod]
        public void TotalErrorClassificationMetric_Error()
        {
            var predictions = new double[] { 0, 1, 1, 2, 3, 4, 4 };
            var targets = new double[] { 0, 1, 1, 2, 2, 3, 4 };

            var sut = new TotalErrorClassificationMetric();
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.2857142857142857, actual);
        }

        [TestMethod]
        public void TotalErrorClassificationMetric_Error_Zero_Error()
        {
            var predictions = new double[] { 0, 1, 1, 2, 3, 4, 4 };
            var targets = new double[] { 0, 1, 1, 2, 3, 4, 4 };

            var sut = new TotalErrorClassificationMetric();
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void TotalErrorClassificationMetric_Error_All_Error()
        {
            var predictions = new double[] { 1, 1, 1, 1 };
            var targets = new double[] { 0, 0, 0, 0 };

            var sut = new TotalErrorClassificationMetric();
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(1.0, actual);
        }

        [TestMethod]
        public void TotalErrorClassificationMetric_ErrorString()
        {
            var predictions = new double[] { 0, 1, 1, 2, 3, 4, 4 };
            var targets = new double[] { 0, 1, 1, 2, 2, 3, 4 };

            var sut = new TotalErrorClassificationMetric();
            var actual = sut.ErrorString(targets, predictions);

            Assert.AreEqual(ExpectedStringResult, actual);
        }

        readonly string ExpectedStringResult = ";0;1;2;3;4;0;1;2;3;4\r\n0;1.00;0.00;0.00;0.00;0.00;1.00;0.00;0.00;0.00;0.00\r\n1;0.00;2.00;0.00;0.00;0.00;0.00;1.00;0.00;0.00;0.00\r\n2;0.00;0.00;1.00;1.00;0.00;0.00;0.00;0.50;0.50;0.00\r\n3;0.00;0.00;0.00;0.00;1.00;0.00;0.00;0.00;0.00;1.00\r\n4;0.00;0.00;0.00;0.00;1.00;0.00;0.00;0.00;0.00;1.00\r\nError: 0.28571\r\n";
    }
}
