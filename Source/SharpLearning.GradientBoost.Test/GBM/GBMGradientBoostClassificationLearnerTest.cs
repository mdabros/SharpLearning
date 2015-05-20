using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.GradientBoost.GBM;
using SharpLearning.Metrics.Regression;
using SharpLearning.Metrics.Classification;
using System.Diagnostics;
using System.Linq;

namespace SharpLearning.GradientBoost.Test.GBM
{
    [TestClass]
    public class GBMGradientBoostClassificationLearnerTest
    {
        [TestMethod]
        public void GBMGradientBoostClassificationLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows("AptitudeTestScore", "PreviousExperience_month").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new GBMGradientBoostClassificationLearner(50, 0.1, 3, 1, 1e-6, 1.0, new GBMBinomialLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, actual);
        }

        [TestMethod]
        public void GBMGradientBoostClassificationLearner_Stochastic_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows("AptitudeTestScore", "PreviousExperience_month").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new GBMGradientBoostClassificationLearner(50, 0.1, 3, 1, 1e-6, .3, new GBMBinomialLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.076923076923076927, actual);
        }

        [TestMethod]
        public void GBMGradientBoostClassificationLearner_MultiClass_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(r => r != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new GBMGradientBoostClassificationLearner(30, 0.1, 3, 1, 1e-6, 1.0, new GBMBinomialLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.018691588785046728, actual);
        }

        [TestMethod]
        public void GBMGradientBoostClassificationLearner_MultiClass_Stochastic_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(r => r != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new GBMGradientBoostClassificationLearner(30, 0.1, 3, 1, 1e-6, 0.5, new GBMBinomialLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.046728971962616821, actual);
        }
    }
}
