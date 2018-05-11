using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.XGBoost.Learners;
using SharpLearning.XGBoost.Models;
using SharpLearning.XGBoost.Test.Properties;

namespace SharpLearning.XGBoost.Test.Learners
{
    [TestClass]
    public class RegressionXGBoostModelTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void RegressionXGBoostModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var learner = CreateLearner();
            using (var sut = learner.Learn(observations, targets))
            {
                var predictions = new double[rows];
                for (int i = 0; i < rows; i++)
                {
                    predictions[i] = sut.Predict(observations.Row(i));
                }

                var evaluator = new MeanSquaredErrorRegressionMetric();
                var error = evaluator.Error(targets, predictions);

                Assert.AreEqual(0.091791017398738781, error, m_delta);
            }
        }

        [TestMethod]
        public void RegressionXGBoostModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var learner = CreateLearner();

            using (var sut = learner.Learn(observations, targets))
            {
                var predictions = sut.Predict(observations);

                var evaluator = new MeanSquaredErrorRegressionMetric();
                var error = evaluator.Error(targets, predictions);

                Assert.AreEqual(0.091791017398738781, error, m_delta);
            }
        }

        [TestMethod]
        public void RegressionXGBoostModel_Save_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var learner = CreateLearner();
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);
            var modelFilePath = "model.xgb";

            using (var sutPreSave = learner.Learn(observations, targets))
            {
                AssertModel(observations, targets, sutPreSave);
                sutPreSave.Save(modelFilePath);
            }

            using (var sutAfterSave = RegressionXGBoostModel.Load(modelFilePath))
            {
                AssertModel(observations, targets, sutAfterSave);
            }
        }

        static RegressionXGBoostLearner CreateLearner()
        {
            return new RegressionXGBoostLearner(maximumTreeDepth: 3,
                learningRate: 0.1,
                estimators: 100,
                silent: true,
                objective: Objective.LinearRegression,
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

        void AssertModel(F64Matrix observations, double[] targets, RegressionXGBoostModel model)
        {
            var predictions = model.Predict(observations);
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.091791017398738781, actual, m_delta);
        }
    }
}
