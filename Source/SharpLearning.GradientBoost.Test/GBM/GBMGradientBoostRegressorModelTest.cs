using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.GradientBoost.GBM;
using SharpLearning.GradientBoost.Test.Properties;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.Metrics.Regression;
using System.Linq;

namespace SharpLearning.GradientBoost.Test.GBM
{
    [TestClass]
    public class GBMGradientBoostRegressorModelTest
    {
        [TestMethod]
        public void GBMGradientBoostRegressorModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new GBMGradientBoostRegressorLearner();
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i));
            }

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.045093177702025665, error, 0.0000001);
        }

        [TestMethod]
        public void GBMGradientBoostRegressorModel_Precit_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new GBMGradientBoostRegressorLearner();
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.045093177702025665, error, 0.0000001);
        }

        [TestMethod]
        public void GBMGradientBoostRegressorModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new GBMGradientBoostRegressorLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 72.1682473281495 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void GBMGradientBoostRegressorModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new GBMGradientBoostRegressorLearner();
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 31.124562320186836, 43.127779144563753 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }
    }
}
