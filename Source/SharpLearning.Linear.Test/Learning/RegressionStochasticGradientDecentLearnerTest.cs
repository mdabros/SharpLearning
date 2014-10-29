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

            var sut = new RegressionStochasticGradientDecentLearner(0.001, 5000, 42, 1);
            var model = sut.Learn(observations, targets);

            var metric = new RootMeanSquareRegressionMetric();
            var predictions = model.Predict(observations);
            var actualError = metric.Error(targets, predictions);

            var actualImportances = model.GetRawVariableImportance();
            Assert.AreEqual(64160.5726271859, actualError, 0.001);
            Assert.AreEqual(336739.710490569, actualImportances[0], 0.001);
            Assert.AreEqual(105731.26301175922, actualImportances[1], 0.001);
            Assert.AreEqual(4544.3634625597488, actualImportances[2], 0.001);
        }
    }
}
