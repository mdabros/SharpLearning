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
    public class GBMLeastAbsoluteDeviationGradientBoostRegressorLearnerTest
    {
        [TestMethod]
        public void GBMLeastAbsoluteDeviationGradientBoostRegressorLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, 1.0, new GBMAbsoluteLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);
            
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.03309315166467057, actual);
        }

        [TestMethod]
        public void GBMLeastAbsoluteDeviationGradientBoostRegressorLearner_Learn_Glass_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, .5, new GBMAbsoluteLoss(), 1);

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

            Assert.AreEqual(0.34945359670803455, actual, 0.0001);
        }

        [TestMethod]
        public void GBMLeastAbsoluteDeviationGradientBoostRegressorLearner_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, 1.0, new GBMAbsoluteLoss(), 1);

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

            Assert.AreEqual(0.41382849800313054, actual, 0.0001);
        }

        [TestMethod]
        public void GBMLeastAbsoluteDeviationGradientBoostRegressorLearner_Stochastic_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, .5, new GBMAbsoluteLoss(), 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.033410130877677545, actual, 0.0001);
        }

        [TestMethod]
        public void GBMLeastAbsoluteDeviationGradientBoostRegressorLearner_Stochastic_Learn_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new GBMGradientBoostRegressorLearner(50, 0.1, 3, 1, 1e-6, .5, new GBMAbsoluteLoss(), 1);

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

            Assert.AreEqual(0.34945359670803455, actual, 0.0001);
        }
    }
}
