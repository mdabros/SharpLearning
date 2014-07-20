using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.DecisionTrees.Test.Models
{
    [TestClass]
    public class RegressionDecisionTreeModelTest
    {
        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionDecisionTreeLearner(4, 100, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i)); 
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionDecisionTreeLearner(4, 100, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionDecisionTreeLearner(4, 100, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
            var predictions = sut.Predict(observations, indices);

            var indexedTargets = targets.GetIndices(indices);
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(indexedTargets, predictions);

            Assert.AreEqual(0.023821615502626264, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;
            var featureNameToIndex = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };

            var learner = new RegressionDecisionTreeLearner(4, 100, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "F2", 100.0 }, { "F1", 0.0 } };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;
            var featureNameToIndex = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };

            var learner = new RegressionDecisionTreeLearner(4, 100, 2, 0.1, 42);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.0, 364.56356850440511 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }
    }
}
