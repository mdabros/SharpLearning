using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Learners;
using SharpLearning.Linear.Test.Properties;
using SharpLearning.Metrics.Regression;
using System.IO;

namespace SharpLearning.Linear.Test.Learners
{
    [TestClass]
    public class RegressionStochasticGradientDecentLearnerTest
    {
        [TestMethod]
        public void RegressionStochasticGradientDecentLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();

            var sut = new RegressionStochasticGradientDecentLearner(0.001, 1000, 0.0, 42, 1);
            var model = sut.Learn(observations, targets);

            var metric = new RootMeanSquareRegressionMetric();
            var predictions = model.Predict(observations);
            var actualError = metric.Error(targets, predictions);

            var actualImportances = model.GetRawVariableImportance();
            Assert.AreEqual(63952.594022178237, actualError, 0.001);
            Assert.AreEqual(109234.21577282451, actualImportances[0], 0.001);
            Assert.AreEqual(4436.0076713921026, actualImportances[1], 0.001);
        }

        [TestMethod]
        public void RegressionStochasticGradientDecentLearner_Learn_Regularized()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();

            var sut = new RegressionStochasticGradientDecentLearner(0.001, 1000, 0.1, 42, 1);
            var model = sut.Learn(observations, targets);

            var metric = new RootMeanSquareRegressionMetric();
            var predictions = model.Predict(observations);
            var actualError = metric.Error(targets, predictions);

            var actualImportances = model.GetRawVariableImportance();
            Assert.AreEqual(72394.353518816875, actualError, 0.001);
            Assert.AreEqual(94409.648546437413, actualImportances[0], 0.001);
            Assert.AreEqual(1991.9172881698214, actualImportances[1], 0.001);
        }
    }
}
