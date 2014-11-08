using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.DecisionTrees.Test.Learners
{
    [TestClass]
    public class ClassificationDecisionTreeLearnerTest
    {
        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Aptitude_Depth_100()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Aptitude(100);
            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_1()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Aptitude(1);
            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_5()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Aptitude(5);
            Assert.AreEqual(0.076923076923076927, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Glass_100()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Glass(100);
            Assert.AreEqual(0.0, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_1()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Glass(1);
            Assert.AreEqual(0.5280373831775701, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_5()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Glass(5);
            Assert.AreEqual(0.16355140186915887, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Aptitude_Depth_100_Weight_1()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(100, 1);
            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_1_Weight_1()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(1, 1);
            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_5_Weight_1()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(5, 1);
            Assert.AreEqual(0.076923076923076927, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Glass_100_Weight_1()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(100, 1);
            Assert.AreEqual(0.0, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_1_Weight_1()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(1, 1);
            Assert.AreEqual(0.5280373831775701, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_5_Weight_1()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(5, 1);
            Assert.AreEqual(0.16355140186915887, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Aptitude_Depth_100_Weight_10()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(100, 10);
            Assert.AreEqual(0.076923076923076927, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Aptitude_depth_5_Weight_10()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(5, 10);
            Assert.AreEqual(0.076923076923076927, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Glass_100_Weight_10()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(100, 10);
            Assert.AreEqual(0.070093457943925228, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationDecisionTreeLearner_Learn_Glass_Depth_5_Weight_10()
        {
            var error = ClassificationDecisionTreeLearner_Learn_Glass_Weighted(5, 10);
            Assert.AreEqual(0.14018691588785046, error, 0.0000001);
        }

        double ClassificationDecisionTreeLearner_Learn_Glass(int treeDepth)
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationDecisionTreeLearner(treeDepth, 1, observations.GetNumberOfColumns(), 0.001, 42);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double ClassificationDecisionTreeLearner_Learn_Aptitude(int treeDepth)
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationDecisionTreeLearner(treeDepth, 1, 2, 0.001, 42);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double ClassificationDecisionTreeLearner_Learn_Glass_Weighted(int treeDepth, double weight)
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var weights = targets.Select(v => Weight(v, 1, weight)).ToArray();
            var sut = new ClassificationDecisionTreeLearner(treeDepth, 1, observations.GetNumberOfColumns(), 0.001, 42);
            var model = sut.Learn(observations, targets, weights);

            var predictions = model.Predict(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            Trace.WriteLine(evaluator.ErrorString(targets, predictions));
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double ClassificationDecisionTreeLearner_Learn_Aptitude_Weighted(int treeDepth, double weight)
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var weights = targets.Select(v => Weight(v, 0, weight)).ToArray();
            var sut = new ClassificationDecisionTreeLearner(treeDepth, 1, 2, 0.001, 42);
            var model = sut.Learn(observations, targets, weights);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            Trace.WriteLine(evaluator.ErrorString(targets, predictions));
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        public double Weight(double v, double targetToWeigh, double weight)
        {
            if (v == targetToWeigh)
                return weight;
            return 1.0;
        }
    }
}
