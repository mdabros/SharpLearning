using System.IO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
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
        public void RegressionXGBoostModel_Save_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var learner = new RegressionXGBoostLearner();
            var model = learner.Learn(observations, targets);

            var predictions = model.Predict(observations);
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

        void AssertModel(F64Matrix observations, double[] targets, RegressionXGBoostModel model)
        {
            var predictions = model.Predict(observations);
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.091791017398738781, actual, m_delta);
        }
    }
}
