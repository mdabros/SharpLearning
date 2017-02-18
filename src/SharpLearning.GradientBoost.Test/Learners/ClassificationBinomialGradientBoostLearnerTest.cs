using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System;
using System.IO;
using System.Linq;

namespace SharpLearning.GradientBoost.Test.Learners
{
    [TestClass]
    public class ClassificationBinomialGradientBoostLearnerTest
    {
        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows("AptitudeTestScore", "PreviousExperience_month").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, actual);
        }

        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows("AptitudeTestScore", "PreviousExperience_month").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(30, 0.1, 3, 1, 1e-6, 1.0, 0, false);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.055555555555555552, actual);
        }

        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_Stochastic_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows("AptitudeTestScore", "PreviousExperience_month").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .3, 0, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.076923076923076927, actual);
        }

        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_Stochastic_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows("AptitudeTestScore", "PreviousExperience_month").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(30, 0.1, 3, 1, 1e-6, .5, 0, false);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.055555555555555552, actual);
        }

        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_MultiClass_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(r => r != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(30, 0.1, 3, 1, 1e-6, 1.0, 0, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.018691588785046728, actual);
        }

        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_MultiClass_FeaturesPrSplit_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(r => r != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(30, 0.1, 3, 1, 1e-6, 1.0, 3, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0514018691588785, actual);
        }


        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_MultiClass_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(30, 0.1, 3, 1, 1e-6, 1.0, 0, false);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_MultiClass_Stochastic_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(r => r != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(30, 0.1, 3, 1, 1e-6, 0.5, 0, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.046728971962616821, actual);
        }

        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_MultiClass_Stochastic_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(30, 0.1, 3, 1, 1e-6, 0.5, 0, false);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.013422818791946308, actual);
        }

        [TestMethod]
        public void ClassificationBinomialGradientBoostLearner_MultiClass_Stochastic_FeaturePrSplit_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationBinomialGradientBoostLearner(30, 0.1, 3, 1, 1e-6, 0.5, 3, false);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.033557046979865772, actual);
        }
    }
}
