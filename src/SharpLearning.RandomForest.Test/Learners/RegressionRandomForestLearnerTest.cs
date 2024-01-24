using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;

namespace SharpLearning.RandomForest.Test.Learners
{
    [TestClass]
    public class RegressionRandomForestLearnerTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Aptitude_Trees_1()
        {
            var error = RegressionRandomForestLearner_Learn_Aptitude(1);
            Assert.AreEqual(0.20257456828885406, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Aptitude_Trees_5()
        {
            var error = RegressionRandomForestLearner_Learn_Aptitude(5);
            Assert.AreEqual(0.17380371993526758, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Aptitude_Trees_100()
        {
            var error = RegressionRandomForestLearner_Learn_Aptitude(100);
            Assert.AreEqual(0.15381141277554411, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Aptitude_Trees_100_SubSample()
        {
            var error = RegressionRandomForestLearner_Learn_Aptitude(100, 0.5);
            Assert.AreEqual(0.19914297129556952, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Glass_1()
        {
            var error = RegressionRandomForestLearnerTest_Learn_Glass(1);
            Assert.AreEqual(0.78499957671177412, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Glass_5()
        {
            var error = RegressionRandomForestLearnerTest_Learn_Glass(5);
            Assert.AreEqual(0.32310955457575374, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Glass_100()
        {
            var error = RegressionRandomForestLearnerTest_Learn_Glass(100);
            Assert.AreEqual(0.22701441864756075, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Glass_100_SubSample()
        {
            var error = RegressionRandomForestLearnerTest_Learn_Glass(100, 0.5);
            Assert.AreEqual(0.4773256918653071, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Glass_100_Indices()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new RegressionRandomForestLearner(100, 1, 100, 1, 0.0001, 1.0, 42, false);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.49709813080602938, error, m_delta);
        }

        [TestMethod]
        public void RegressionRandomForestLearnerTest_Learn_Glass_100_Trees_Parallel()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new RegressionRandomForestLearner(100, 1, 100, 1, 0.0001, 1.0, 42, true);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.22701441864756075, error, m_delta);
        }

        double RegressionRandomForestLearnerTest_Learn_Glass(int trees, double subSampleRatio = 1.0)
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = new RegressionRandomForestLearner(trees, 1, 100, 1, 0.0001,
                subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double RegressionRandomForestLearner_Learn_Aptitude(int trees, double subSampleRatio = 1.0)
        {
            var (observations, targets) = DataSetUtilities.LoadAptitudeDataSet();

            var sut = new RegressionRandomForestLearner(trees, 5, 100, 1, 0.0001,
                subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);
            return error;
        }
    }
}
