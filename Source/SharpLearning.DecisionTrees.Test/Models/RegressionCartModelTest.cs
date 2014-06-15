using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.DecisionTrees.Test.Models
{
    [TestClass]
    public class RegressionCartModelTest
    {
        [TestMethod]
        public void RegressionCartModel_Single_Observation()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var sut = new RegressionCartLearner(4, 100, 0.1);
            var model = sut.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = model.Predict(observations.GetRow(i)); 
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038873687234849852, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionCartModel_Single_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var sut = new RegressionCartLearner(4, 100, 0.1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038873687234849852, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionCartModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;
            var featureNameToIndex = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };

            var sut = new RegressionCartLearner(4, 100, 0.1);
            var model = sut.Learn(observations, targets);

            var actual = model.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "F2", 100.0 }, { "F1", 0.0 } };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RegressionCartModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;
            var featureNameToIndex = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };

            var sut = new RegressionCartLearner(4, 100, 0.1);
            var model = sut.Learn(observations, targets);

            var actual = model.GetRawVariableImportance();
            var expected = new double[] { 0.0, 2.070405777290854 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }
    }
}
