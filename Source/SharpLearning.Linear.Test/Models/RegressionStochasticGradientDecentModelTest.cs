using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Learning;
using SharpLearning.Linear.Test.Properties;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.Linear.Test.Models
{
    [TestClass]
    public class RegressionStochasticGradientDecentModelTest
    {
        [TestMethod]
        public void RegressionStochasticGradientDecentLearner_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();

            var learner = new RegressionStochasticGradientDecentLearner(0.001, 1000, 0.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[targets.Length];
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i));
            }

            var metric = new RootMeanSquareRegressionMetric();
            var actual = metric.Error(targets, predictions);
            Assert.AreEqual(63952.594022178237, actual, 0.001);
        }

        [TestMethod]
        public void RegressionStochasticGradientDecentLearner_Predict_Multi()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();

            var sut = new RegressionStochasticGradientDecentLearner(0.001, 1000, 0.0, 42, 1);
            var model = sut.Learn(observations, targets);

            var metric = new RootMeanSquareRegressionMetric();

            var predictions = model.Predict(observations);
            var actual = metric.Error(targets, predictions);
            Assert.AreEqual(63952.594022178237, actual, 0.001);
        }

        [TestMethod]
        public void RegressionStochasticGradientDecentLearner_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();

            var sut = new RegressionStochasticGradientDecentLearner(0.001, 1000, 0.0, 42, 1);
            var model = sut.Learn(observations, targets);

            var actual = model.GetRawVariableImportance();
            Assert.AreEqual(109234.21577282451, actual[0], 0.001);
            Assert.AreEqual(4436.0076713921026, actual[1], 0.001);
        }

        [TestMethod]
        public void RegressionStochasticGradientDecentLearner_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var featureNameToIndex = parser.EnumerateRows("Size", "Rooms").First().ColumnNameToIndex;
            var targets = parser.EnumerateRows("Price").ToF64Vector();

            var sut = new RegressionStochasticGradientDecentLearner(0.001, 1000, 0.0, 42, 1);
            var model = sut.Learn(observations, targets);

            var actual = model.GetVariableImportance(featureNameToIndex).ToList();
            var expected = new Dictionary<string, double> { { "Size", 100.0 }, { "Rooms", 32.1115502727675 } }.ToList();
            
            Assert.AreEqual(expected.Count, actual.Count);
            for (int i = 0; i < expected.Count; i++)
            {
                Assert.AreEqual(expected[i].Key, actual[i].Key);
                Assert.AreEqual(expected[i].Value, actual[i].Value, 0.0001);
            }
        }
    }
}
