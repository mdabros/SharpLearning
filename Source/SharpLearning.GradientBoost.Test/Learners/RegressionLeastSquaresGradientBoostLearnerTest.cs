using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System;
using System.IO;
using System.Linq;

namespace SharpLearning.GradientBoost.Test.Learners
{
    [TestClass]
    public class RegressionLeastSquaresGradientBoostLearnerTest
    {
        [TestMethod]
        public void RegressionLeastSquaresGradientBoostLearner_Learn_AptitudeData()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new RegressionLeastSquaresGradientBoostLearner();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.052593937296318408, actual);
        }

        [TestMethod]
        public void RegressionLeastSquaresGradientBoostLearner_Stochastic_Learn_AptitudeData()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new RegressionLeastSquaresGradientBoostLearner(100, 0.1, 3, 2000, 1, 0.000001, 0.5);

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.278018565057251, actual);
        }

        [TestMethod]
        public void RegressionLeastSquaresGradientBoostLearner_Learn_Glass()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionLeastSquaresGradientBoostLearner();

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.18457759227201861, actual);
        }

        [TestMethod]
        public void RegressionLeastSquaresGradientBoostLearner_Learn_Glass_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionLeastSquaresGradientBoostLearner();

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.11971320408918414, actual);
        }

    }
}
