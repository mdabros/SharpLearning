using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learning;
using SharpLearning.AdaBoost.Test.Properties;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.AdaBoost.Test.Models
{
    [TestClass]
    public class RegressionAdaBoostModelTest
    {
        [TestMethod]
        public void RegressionAdaBoostModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i));
            }

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.14185814185814186, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionAdaBoostModel_Precit_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.14185814185814186, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionAdaBoostModel_Precit_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);
            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };

            var predictions = sut.Predict(observations, indices);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var indexedTargets = targets.GetIndices(indices);
            var error = evaluator.Error(indexedTargets, predictions);

            Assert.AreEqual(0.14491341991341991, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionAdaBoostModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 6.08463169539965 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void RegressionAdaBoostModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 1.2068886913382497, 19.835032780221198 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }
    }
}
