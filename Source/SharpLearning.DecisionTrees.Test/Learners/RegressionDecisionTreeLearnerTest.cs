using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.Metrics.Regression;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.DecisionTrees.Test.Learners
{
    [TestClass]
    public class RegressionDecisionTreeLearnerTest
    {
        [TestMethod]
        public void RegressionDecisionTreeLearner_Learn_Depth_100()
        {
            var error = RegressionDecisionTreeLearner_Learn(100);
            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeLearner_Learn_Depth_1()
        {
            var error = RegressionDecisionTreeLearner_Learn(1);
            Assert.AreEqual(0.55139468272009107, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeLearner_Learn_Depth_3()
        {
            var error = RegressionDecisionTreeLearner_Learn(2);
            Assert.AreEqual(0.14322350107327153, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeLearner_Learn_Depth_100_Weight_1()
        {
            var error = RegressionDecisionTreeLearner_Learn_Weighted(100, 1.0);
            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeLearner_Learn_Depth_1_Weight_1()
        {
            var error = RegressionDecisionTreeLearner_Learn_Weighted(1, 1);
            Assert.AreEqual(0.55139468272009107, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeLearner_Learn_Depth_3_Weight_1()
        {
            var error = RegressionDecisionTreeLearner_Learn_Weighted(2, 1);
            Assert.AreEqual(0.14322350107327153, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeLearner_Learn_Depth_100_Weight_100()
        {
            var error = RegressionDecisionTreeLearner_Learn_Weighted(100, 100.0);
            Assert.AreEqual(0.032256921590414704, error, 0.0000001);
        }

        private static double RegressionDecisionTreeLearner_Learn(int treeDepth)
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var sut = new RegressionDecisionTreeLearner(treeDepth, 4, 2, 0.1, 42);

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        private double RegressionDecisionTreeLearner_Learn_Weighted(int treeDepth, double weight)
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var sut = new RegressionDecisionTreeLearner(treeDepth, 4, 2, 0.1, 42);
            var weights = targets.Select(v => Weight(v, weight)).ToArray();
            var model = sut.Learn(observations, targets, weights);

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double Weight(double v, double weight)
        {
            if (v < 3.0)
                return weight;
            return 1.0;
        }
    }
}
