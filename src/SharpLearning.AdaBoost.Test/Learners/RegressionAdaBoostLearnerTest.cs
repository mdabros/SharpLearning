using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.Containers.Extensions;
using SharpLearning.Metrics.Regression;

namespace SharpLearning.AdaBoost.Test.Learners
{
    [TestClass]
    public class RegressionAdaBoostLearnerTest
    {
        [TestMethod]
        public void RegressionAdaBoostLearner_Learn_AptitudeData_LinearLoss()
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

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
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

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
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

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
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

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
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

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
    }
}
