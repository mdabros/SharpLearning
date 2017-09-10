using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.AdaBoost.Test.Properties;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Containers.Extensions;
using System;
using System.IO;
using System.Linq;
using System.Diagnostics;

namespace SharpLearning.AdaBoost.Test.Learners
{
    [TestClass]
    public class RegressionAdaBoostLearnerTest
    {
        [TestMethod]
        public void RegressionAdaBoostLearner_Learn_AptitudeData_LinearLoss()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new RegressionAdaBoostLearner(10);
            
            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.14185814185814186, actual);
        }

        [TestMethod]
        public void RegressionAdaBoostLearner_Learn_AptitudeData_SquaredLoss()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new RegressionAdaBoostLearner(10, 1, 0, AdaBoostRegressionLoss.Squared);

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.13672161172161174, actual);
        }

        [TestMethod]
        public void RegressionAdaBoostLearner_Learn_AptitudeData_ExponentialLoss()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new RegressionAdaBoostLearner(10, 1, 0, AdaBoostRegressionLoss.Exponential);

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.10370879120879124, actual);
        }

        [TestMethod]
        public void RegressionAdaBoostLearner_Learn_Glass()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionAdaBoostLearner(10);

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.54723570404775324, actual);
        }

        [TestMethod]
        public void RegressionAdaBoostLearner_Learn_Glass_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionAdaBoostLearner(10, 1, 5);

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

            Assert.AreEqual(0.22181054803405248, actual);
        }

        [TestMethod]
        public void RegressionAdaBoostLearner_Learn_Glass_Weighted()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var classSizes = targets.GroupBy(v => v).ToDictionary(v => v.Key, v => v.Count());
            var weights = targets.Select(v => (double)targets.Length / (double)classSizes[v]).ToArray(); // balanced
            //var weights = targets.Select(v => v == 6.0 ? 10.0 : 1.0).ToArray(); // specific
            //var weights = targets.Select(v => v > 4 ? 10.0 : 1.0).ToArray(); // prioritize large targets

            var sut = new RegressionAdaBoostLearner(10);

            var model = sut.Learn(observations, targets, weights);
            var predictions = model.Predict(observations);

            var targetPredictions = targets.Zip(predictions, (t, p) => new { Target = t, Prediction = p })
                .GroupBy(v => v.Target)
                .ToArray();

            var evaluator = new MeanSquaredErrorRegressionMetric();

            foreach (var pair in targetPredictions)
            {
                Trace.WriteLine($"Target: {pair.Key}, Samples: {pair.Count()},  Error: {evaluator.Error(pair.Select(v => v.Target).ToArray(), pair.Select(v => v.Prediction).ToArray())}");
            }

            Trace.WriteLine($"Overall Error: {evaluator.Error(targets, predictions)}");

            var actual = evaluator.Error(targets, predictions);
            Assert.AreEqual(0.50901425532981959, actual, 0.0001);
        }

    }
}
