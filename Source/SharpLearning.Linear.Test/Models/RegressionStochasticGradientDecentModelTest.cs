using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Learners;
using SharpLearning.Linear.Models;
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

        [TestMethod]
        public void RegressionStochasticGradientDecentLearner_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();

            var learner = new RegressionStochasticGradientDecentLearner(0.001, 1000, 0.0, 42, 1);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);
            var actual = writer.ToString();
            Assert.AreEqual(RegressionStochasticGradientDecentModelString, actual);
        }

        [TestMethod]
        public void RegressionStochasticGradientDecentLearner_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Housing));
            var observations = parser.EnumerateRows("Size", "Rooms").ToF64Matrix();
            var targets = parser.EnumerateRows("Price").ToF64Vector();

            var reader = new StringReader(RegressionStochasticGradientDecentModelString);
            var sut = RegressionStochasticGradientDecentModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var metric = new RootMeanSquareRegressionMetric();
            var actual = metric.Error(targets, predictions);
            Assert.AreEqual(63952.594022178237, actual, 0.001);
        }

        readonly string RegressionStochasticGradientDecentModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionStochasticGradientDecentModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.Linear.Models\">\r\n  <m_weights xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"2\" z:Size=\"3\">\r\n    <d2p1:double>340171.1061750318</d2p1:double>\r\n    <d2p1:double>109234.21577282451</d2p1:double>\r\n    <d2p1:double>-4436.0076713921026</d2p1:double>\r\n  </m_weights>\r\n</RegressionStochasticGradientDecentModel>";

    }
}
