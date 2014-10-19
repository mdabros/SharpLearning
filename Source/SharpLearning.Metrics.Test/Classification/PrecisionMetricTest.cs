using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Classification;
using System;

namespace SharpLearning.Metrics.Test.Classification
{
    [TestClass]
    public class PrecisionMetricTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void PrecisionMetric_Not_Binary()
        {
            var targets = new double[] { 0, 1, 1, 2 };
            var predictions = new double[] { 0, 1, 1, 2 };

            var sut = new PrecisionMetric<double>(1);
            var actual = sut.Error(targets, predictions);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void PrecisionMetric_Not_Different_Lengths()
        {
            var targets = new double[] { 0, 1, 1, 1 };
            var predictions = new double[] { 0, 1, 1, 1, 0 };

            var sut = new PrecisionMetric<double>(1);
            var actual = sut.Error(targets, predictions);
        }

        [TestMethod]
        public void PrecisionMetric_No_Error()
        {
            var targets = new double[] { 0, 1, 1 };
            var predictions = new double[] { 0, 1, 1 };

            var sut = new PrecisionMetric<double>(1);
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void PrecisionMetric_All_Error()
        {
            var targets = new double[] { 0, 1, 0 };
            var predictions = new double[] { 1, 0, 1 };

            var sut = new PrecisionMetric<double>(1);
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(1.0, actual);
        }

        [TestMethod]
        public void PrecisionMetric_Error()
        {
            var targets = new double[] { 0, 1, 1, 1, 1, 0, 0, 1};
            var predictions = new double[] { 1, 1, 1, 0, 0, 0, 1, 1 };

            var sut = new PrecisionMetric<double>(1);
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.40000000000000002, actual);
        }

        [TestMethod]
        public void PrecisionMetric_Error_All_Negative()
        {
            var targets = new double[] { 0, 0, 0, 0, 0, 0, 0, 1 };
            var predictions = new double[] { 0, 0, 0, 0, 0, 0, 0, 0 };

            var sut = new PrecisionMetric<double>(1);
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(1.0, actual);
        }

        [TestMethod]
        public void PrecisionMetric_Error_All_Positive()
        {
            var targets = new double[] { 0, 0, 0, 0, 0, 0, 0, 1 };
            var predictions = new double[] { 1, 1, 1, 1, 1, 1, 1, 1 };

            var sut = new PrecisionMetric<double>(1);
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.875, actual);
        }

        [TestMethod]
        public void PrecisionMetric_ErrorString()
        {
            var targets = new double[] { 0, 1, 0 };
            var predictions = new double[] { 1, 0, 1 };

            var sut = new PrecisionMetric<double>(1);
            var actual = sut.ErrorString(targets, predictions);
            var expected = ";0;1;0;1\r\n0;0.00;2.00;0.00;1.00\r\n1;1.00;0.00;1.00;0.00\r\nError: 1.00000\r\n";

            Assert.AreEqual(expected, actual);
        }
    }
}
