using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.CrossValidation.Test;
using SharpLearning.CrossValidation.Test.Properties;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System;
using System.IO;
using System.Linq;

namespace SharpLearning.CrossValidation.CrossValidators.Test
{
    [TestClass]
    public class NoShuffleCrossValidationTest
    {
        [TestMethod]
        public void NoShuffleCrossValidation_CrossValidate_Folds_2()
        {
            var actual = CrossValidate(2);
            Assert.AreEqual(0.08399278971163, actual, 0.001);
        }

        [TestMethod]
        public void NoShuffleCrossValidation_CrossValidate_Folds_10()
        {
            var actual = CrossValidate(10);
            Assert.AreEqual(0.069356782107075, actual, 0.001);
        }

        [TestMethod]
        public void NoShuffleCrossValidation_CrossValidate_Provide_Indices_Folds_2()
        {
            var actual = CrossValidate_Provide_Indices(2);
            Assert.AreEqual(0.10219923025847003, actual, 0.001);
        }

        [TestMethod]
        public void NoShuffleCrossValidation_CrossValidate_Provide_Indices_Folds_10()
        {
            var actual = CrossValidate_Provide_Indices(10);
            Assert.AreEqual(0.10969696400057005, actual, 0.001);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void NoShuffleCrossValidation_CrossValidate_Too_Many_Folds()
        {
            CrossValidate(2000);
        }

        double CrossValidate(int folds)
        {
            var targetName = "T";
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();

            var sut = new NoShuffleCrossValidation<double>(folds);
            var predictions = sut.CrossValidate(new RegressionDecisionTreeLearner(), observations, targets);
            var metric = new MeanSquaredErrorRegressionMetric();

            return metric.Error(targets, predictions);
        }

        double CrossValidate_Provide_Indices(int folds)
        {
            var targetName = "T";
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows(v => !v.Contains(targetName)).ToF64Matrix();
            var targets = parser.EnumerateRows(targetName).ToF64Vector();

            var sut = new NoShuffleCrossValidation<double>(folds);

            var rowsToCrossvalidate = targets.Length / 2;
            var indices = Enumerable.Range(0, rowsToCrossvalidate).ToArray();
            var predictions = new double[rowsToCrossvalidate];
            
            sut.CrossValidate(new RegressionDecisionTreeLearner(), observations, targets, indices, predictions);
            var metric = new MeanSquaredErrorRegressionMetric();

            return metric.Error(targets.Take(rowsToCrossvalidate).ToArray(), predictions);
        }
    }
}
