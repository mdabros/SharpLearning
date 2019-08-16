using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Metrics.Classification;

namespace SharpLearning.Metrics.Test.Classification
{
    [TestClass]
    public class LogLossClassificationProbabilityMetricTest
    {
        [TestMethod]
        public void LogLossClassificationMetric_Error_1()
        {
            var sut = new LogLossClassificationProbabilityMetric(1e-15);
            var predictions = new ProbabilityPrediction[] { 
                new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1.0 }, { 1, 0.0 }, { 2, 0.0 } }),
                new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0 }, { 1, 1.0 }, { 2, 0.0 } }),
                new ProbabilityPrediction(2, new Dictionary<double, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 1.0 } }),
            };

            var targets = new double[] { 0, 1, 2 };

            var actual = sut.Error(targets, predictions);
            Assert.AreEqual(9.9920072216264128e-16, actual, 1e-17);
        }

        [TestMethod]
        public void LogLossClassificationMetric_Error_2()
        {
            var sut = new LogLossClassificationProbabilityMetric(1e-15);
            var predictions = new ProbabilityPrediction[] { 
                new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } }),
                new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0 }, { 1, 1.0 }, { 2, 0.0 } }),
                new ProbabilityPrediction(2, new Dictionary<double, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 1.0 } }),
            };

            var targets = new double[] { 0, 1, 2 };

            var actual = sut.Error(targets, predictions);
            Assert.AreEqual(0.36620409622270467, actual, 0.0001);
        }

        [TestMethod]
        public void LogLossClassificationMetric_ErrorString()
        {
            var sut = new LogLossClassificationProbabilityMetric(1e-15);
            var predictions = new ProbabilityPrediction[] { 
                new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } }),
                new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0 }, { 1, 1.0 }, { 2, 0.0 } }),
                new ProbabilityPrediction(2, new Dictionary<double, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 1.0 } }),
            };

            var targets = new double[] { 0, 1, 2 };

            var actual = sut.ErrorString(targets, predictions);
            var expected = ";0;1;2;0;1;2\r\n0;1.000;0.000;0.000;100.000;0.000;0.000\r\n1;0.000;1.000;0.000;0.000;100.000;0.000\r\n2;0.000;0.000;1.000;0.000;0.000;100.000\r\nError: 36.620\r\n";
            
            Assert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void LogLossClassificationMetric_ErrorString_TargetStringMapping()
        {
            var sut = new LogLossClassificationProbabilityMetric(1e-15);
            var predictions = new ProbabilityPrediction[] { 
                new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 1.0 }, { 1, 1.0 }, { 2, 1.0 } }),
                new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.0 }, { 1, 1.0 }, { 2, 0.0 } }),
                new ProbabilityPrediction(2, new Dictionary<double, double> { { 0, 0.0 }, { 1, 0.0 }, { 2, 1.0 } }),
            };

            var targets = new double[] { 0, 1, 2 };
            var targetStringMapping = new Dictionary<double, string> { { 0, "One" }, { 1, "Two" }, { 2, "Three" } };

            var actual = sut.ErrorString(targets, predictions, targetStringMapping);
            var expected = ";One;Two;Three;One;Two;Three\r\nOne;1.000;0.000;0.000;100.000;0.000;0.000\r\nTwo;0.000;1.000;0.000;0.000;100.000;0.000\r\nThree;0.000;0.000;1.000;0.000;0.000;100.000\r\nError: 36.620\r\n";

            Assert.AreEqual(expected, actual);
        }
    }
}
