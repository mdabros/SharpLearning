using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Learning;
using SharpLearning.Linear.Test.Properties;
using SharpLearning.Metrics.Regression;
using System.IO;

namespace SharpLearning.Linear.Test.Learning
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

            var sut = new RegressionStochasticGradientDecentLearner(0.001, 1000, 42, 1);
            var model = sut.Learn(observations, targets);

            var metric = new RootMeanSquareRegressionMetric();
            var predictions = model.Predict(observations);
            var actualError = metric.Error(targets, predictions);

            var actualImportances = model.GetRawVariableImportance();
            Assert.AreEqual(63952.594022178237, actualError, 0.001);
            Assert.AreEqual(109234.21577282451, actualImportances[0], 0.001);
            Assert.AreEqual(4436.0076713921026, actualImportances[1], 0.001);
        }
    }
}
