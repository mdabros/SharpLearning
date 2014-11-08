using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Linear.Learning;
using SharpLearning.Linear.Test.Properties;
using SharpLearning.Metrics.Classification;
using SharpLearning.Metrics.Regression;
using System.IO;

namespace SharpLearning.Linear.Test.Learning
{
    [TestClass]
    public class ClassificationStochasticGradientDecentLearnerTest
    {
        [TestMethod]
        public void ClassificationStochasticGradientDecentLearner_Learn_Binary()
        {
            var parser = new CsvParser(() => new StringReader(Resources.LogitTest));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationStochasticGradientDecentLearner(0.0001, 100000, 42, 1);
            var model = sut.Learn(observations, targets);

            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(observations);
            var actualError = metric.Error(targets, predictions);

            var actualImportances = model.GetRawVariableImportance();
            Assert.AreEqual(0.11, actualError, 0.001);
            Assert.AreEqual(0.14011316001536858, actualImportances[0], 0.001);
            Assert.AreEqual(0.13571128043372779, actualImportances[1], 0.001);
        }

        [TestMethod]
        public void ClassificationStochasticGradientDecentLearner_Learn_Multi()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(s => s != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationStochasticGradientDecentLearner(0.1, 750, 42, 1);
            var model = sut.Learn(observations, targets);

            var metric = new TotalErrorClassificationMetric<double>();
            var predictions = model.Predict(observations);
            var actualError = metric.Error(targets, predictions);

            Assert.AreEqual(0.46728971962616822, actualError, 0.001);
        }
    }
}
