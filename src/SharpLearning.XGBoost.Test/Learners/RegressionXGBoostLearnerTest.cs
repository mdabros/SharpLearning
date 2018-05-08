using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.XGBoost.Learners;
using SharpLearning.XGBoost.Test.Properties;

namespace SharpLearning.XGBoost.Test.Learners
{
    [TestClass]
    public class RegressionXGBoostLearnerTest
    {
        readonly double m_delta = 0.0000001;

        [TestMethod]
        public void RegressionXGBoostLearner_Learn()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new RegressionXGBoostLearner();
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.091791017398738781, error, m_delta);
        }
    }
}
