using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.Metrics.Test.Regression
{
    [TestClass]
    public class DiscreteTargetMeanErrorRegressionMetricTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentNullException))]
        public void DiscreteTargetMeanErrorRegressionMetric_Internal_Metric_IsNull()
        {
            new DiscreteTargetMeanErrorRegressionMetric(null);
        }

        [TestMethod]
        public void DiscreteTargetMeanErrorRegressionMetric_Error_Zero_Error()
        {
            var targets = new double[] { 0, 0, 0, 0, 0, 0 };
            var predictions = new double[] { 0, 0, 0, 0, 0, 0 };

            var sut = new DiscreteTargetMeanErrorRegressionMetric();
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void DiscreteTargetMeanErrorRegressionMetric_Error()
        {
            var targets = new double[] { 1.0, 1.0, 2.0, 3.0, 3.0 };
            var predictions = new double[] { 0.8, 1.2, 1.5, 1.8, 3.9 };

            var sut = new DiscreteTargetMeanErrorRegressionMetric();
            var actual = sut.Error(targets, predictions);

            Assert.AreEqual(0.47166666666666668, actual, 0.00001);
        }
    }
}
