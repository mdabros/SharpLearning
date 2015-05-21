using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.GradientBoost.GBM;
using SharpLearning.Metrics.Regression;
using System.Linq;
using SharpLearning.Containers.Extensions;

namespace SharpLearning.GradientBoost.Test.GBM
{
    [TestClass]
    public class GBMGradientBoostRegressorLearnerTest
    {
        [TestMethod]
        public void GBMGradientBoostRegressorLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, 1.0, new GBMSquaredLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);
            
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.023436850973295304, actual);
        }

        [TestMethod]
        public void GBMGradientBoostRegressorLearner_Learn_Glass_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, .5, new GBMSquaredLoss(), 1);

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

            Assert.AreEqual(0.29330180231264918, actual, 0.0001);
        }

        [TestMethod]
        public void GBMGradientBoostRegressorLearner_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, 1.0, new GBMSquaredLoss(), 1);

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

            Assert.AreEqual(0.23625469946001074, actual, 0.0001);
        }

        [TestMethod]
        public void GBMGradientBoostRegressorLearner_Stochastic_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, .5, new GBMSquaredLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.025391913155163696, actual, 0.0001);
        }

        [TestMethod]
        public void GBMGradientBoostRegressorLearner_Stochastic_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, .5, new GBMSquaredLoss(), 1);

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

            Assert.AreEqual(0.29330180231264918, actual, 0.0001);
        }
    }
}
