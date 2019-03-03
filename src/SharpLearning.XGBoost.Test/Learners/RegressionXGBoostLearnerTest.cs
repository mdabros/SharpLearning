using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Extensions;
using SharpLearning.Metrics.Regression;
using SharpLearning.XGBoost.Learners;

namespace SharpLearning.XGBoost.Test.Learners
{
    [TestClass]
    public class RegressionXGBoostLearnerTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void RegressionXGBoostLearner_Learn()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var sut = CreateLearner();

            using (var model = sut.Learn(observations, targets))
            {
                var predictions = model.Predict(observations);

                var evaluator = new MeanSquaredErrorRegressionMetric();
                var error = evaluator.Error(targets, predictions);

                Assert.AreEqual(0.0795934933096642, error, m_delta);
            }
        }

        [TestMethod]
        public void RegressionXGBoostLearner_Learn_Indexed()
        {
            var (observations, targets) = DataSetUtilities.LoadGlassDataSet();

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var sut = CreateLearner();

            using (var model = sut.Learn(observations, targets, indices))
            {
                var predictions = model.Predict(observations);

                var evaluator = new MeanSquaredErrorRegressionMetric();
                var error = evaluator.Error(targets, predictions);

                Assert.AreEqual(0.33589853954956522, error, m_delta);
            }
        }

        static RegressionXGBoostLearner CreateLearner()
        {
            return new RegressionXGBoostLearner(maximumTreeDepth: 3,
                learningRate: 0.1,
                estimators: 100,
                silent: true,
                objective: RegressionObjective.LinearRegression,
                boosterType: BoosterType.GBTree,
                treeMethod: TreeMethod.Auto,
                numberOfThreads: -1,
                gamma: 0,
                minChildWeight: 1,
                maxDeltaStep: 0,
                subSample: 1,
                colSampleByTree: 1,
                colSampleByLevel: 1,
                l1Regularization: 0,
                l2Reguralization: 1,
                scalePosWeight: 1,
                baseScore: 0.5,
                seed: 0,
                missing: double.NaN);
        }
    }
}
