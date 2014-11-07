using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Learning;
using SharpLearning.Linear.Test.Properties;
using SharpLearning.Metrics.Classification;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpLearning.Linear.Test.Models
{
    [TestClass]
    public class ClassificationStochasticGradientDecentModelTest
    {
        [TestMethod]
        public void ClassificationStochasticGradientDecentLearner_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var learner = new ClassificationStochasticGradientDecentLearner(0.0001, 10000000, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[targets.Length];
            for (int i = 0; i < predictions.Length; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i));
            }

            var metric = new TotalErrorClassificationMetric<double>();
            var actual = metric.Error(targets, predictions);
            Assert.AreEqual(0.08, actual, 0.001);
        }

        [TestMethod]
        public void ClassificationStochasticGradientDecentLearner_Predict_Multi()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationStochasticGradientDecentLearner(0.0001, 10000000, 42, 1);
            var model = sut.Learn(observations, targets);

            var metric = new TotalErrorClassificationMetric<double>();

            var predictions = model.Predict(observations);
            var actual = metric.Error(targets, predictions);
            Assert.AreEqual(0.08, actual, 0.001);
        }

        [TestMethod]
        public void ClassificationStochasticGradientDecentLearner_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationStochasticGradientDecentLearner(0.0001, 10000000, 42, 1);
            var model = sut.Learn(observations, targets);

            var actual = model.GetRawVariableImportance();
            Assert.AreEqual(15.934288283579136, actual[0], 0.001);
            Assert.AreEqual(0.14011316001536858, actual[1], 0.001);
            Assert.AreEqual(0.13571128043372779, actual[2], 0.001);
        }

        [TestMethod]
        public void ClassificationStochasticGradientDecentLearner_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var featureNameToIndex = parser.EnumerateRows("F1", "F2").First().ColumnNameToIndex;
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationStochasticGradientDecentLearner(0.0001, 10000000, 42, 1);
            var model = sut.Learn(observations, targets);

            var actual = model.GetVariableImportance(featureNameToIndex).ToList();
            var expected = new Dictionary<string, double> { { "F1", 100.0 }, { "F2", 0.879318595984989 } }.ToList();
            
            Assert.AreEqual(expected.Count, actual.Count);
            for (int i = 0; i < expected.Count; i++)
            {
                Assert.AreEqual(expected[i].Key, actual[i].Key);
                Assert.AreEqual(expected[i].Value, actual[i].Value, 0.0001);
            }
        }
    }
}
