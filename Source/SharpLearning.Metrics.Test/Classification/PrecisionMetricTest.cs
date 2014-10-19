using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Metrics.Test.Classification
{
    [TestClass]
    public class PrecisionMetricTest
    {
        [TestMethod]
        public void PrecisionMetric_No_Error()
        {
            var targets = new double[] { 0, 1, 2 };
            var predictions = new double[] { 0, 1, 2 };

            var sut = new PrecisionMetric<double>();
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void PrecisionMetric_All_Error()
        {
            var targets = new double[] { 0, 1, 2 };
            var predictions = new double[] { 2, 2, 1 };

            var sut = new PrecisionMetric<double>();
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(1.0, actual);
        }

        [TestMethod]
        public void PrecisionMetric_Error()
        {
            var targets = new double[] { 0, 1, 2, 2, 2, 3, 3, 1};
            var predictions = new double[] { 1, 1, 2, 3, 2, 3, 3, 1 };

            var sut = new PrecisionMetric<double>();
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.25, actual);
        }
    }
}
