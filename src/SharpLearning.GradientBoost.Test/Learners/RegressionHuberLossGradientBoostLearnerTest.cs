using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.GradientBoost.Test.Learners
{
    [TestClass]
    public class RegressionHuberLossGradientBoostLearnerTest
    {
        [TestMethod]
        public void RegressionHuberLossGradientBoostLearner_Learn()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var sut = new RegressionHuberLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, 0.9, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.025062685962747234, actual);
        }

        [TestMethod]
        public void RegressionHuberLossGradientBoostLearner_FeaturesPrSplit_Learn()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var sut = new RegressionHuberLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 1, 0.9, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.076395719114033783, actual);
        }

        [TestMethod]
        public void RegressionHuberLossGradientBoostLearner_Learn_Glass_Indexed()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new RegressionHuberLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, 0.9, false);

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

            Assert.AreEqual(0.27897874395064975, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionHuberLossGradientBoostLearner_Learn_Indexed()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new RegressionHuberLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, 1.0, 0, 0.9, false);

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

            Assert.AreEqual(0.21635632980552189, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionHuberLossGradientBoostLearner_Stochastic_Learn()
        {
            var (observations, targets) = DataSetUtilities.LoadDecisionTreeDataSet();

            var sut = new RegressionHuberLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, 0.9, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0268131680885004, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionHuberLossGradientBoostLearner_Stochastic_Learn_Indexed()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new RegressionHuberLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 0, 0.9, false);

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

            Assert.AreEqual(0.27897874395064975, actual, 0.0001);
        }

        [TestMethod]
        public void RegressionHuberLossGradientBoostLearner_Stochastic_FeaturesPrSplit_Learn_Indexed()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new RegressionHuberLossGradientBoostLearner(50, 0.1, 3, 1, 1e-6, .5, 3, 0.9, false);

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

            Assert.AreEqual(0.33489277579485843, actual, 0.0001);
        }
    }
}
