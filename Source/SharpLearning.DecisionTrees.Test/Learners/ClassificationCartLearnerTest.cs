using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System;
using System.Diagnostics;
using System.IO;

namespace SharpLearning.DecisionTrees.Test.Learners
{
    [TestClass]
    public class ClassificationCartLearnerTest
    {
        [TestMethod]
        public void ClassificationCartLearner_Learn_Aptitude_Depth_100()
        {
            var error = ClassificationCartLearner_Learn_Aptitude(100);
            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationCartLearner_Learn_Aptitude_depth_1()
        {
            var error = ClassificationCartLearner_Learn_Aptitude(1);
            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationCartLearner_Learn_Aptitude_depth_5()
        {
            var error = ClassificationCartLearner_Learn_Aptitude(5);
            Assert.AreEqual(0.076923076923076927, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationCartLearner_Learn_Glass_100()
        {
            var error = ClassificationCartLearner_Learn_Glass(100);
            Assert.AreEqual(0.0, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationCartLearner_Learn_Glass_Depth_1()
        {
            var error = ClassificationCartLearner_Learn_Glass(1);
            Assert.AreEqual(0.5280373831775701, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationCartLearner_Learn_Glass_Depth_5()
        {
            var error = ClassificationCartLearner_Learn_Glass(5);
            Assert.AreEqual(0.16822429906542055, error, 0.0000001);
        }

        double ClassificationCartLearner_Learn_Glass(int treeDepth)
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationCartLearner(1, treeDepth, observations.GetNumberOfColumns(), 0.001);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double ClassificationCartLearner_Learn_Aptitude(int treeDepth)
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationCartLearner(1, treeDepth, 2, 0.001);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        [Ignore]
        [TestMethod]
        public void ClassificationCartLearner_Learn_Timing()
        {
            var rows = 10000;
            var cols = 10;

            var random = new Random(42);
            var observations = new F64Matrix(rows, cols);
            var targets = new double[rows];
            
            for (int i = 0; i < targets.Length; i++)
            {
                targets[i] = random.Next(5);
            }
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    observations.SetItemAt(i, j, random.NextDouble());   
                }
            }

            var sut = new ClassificationCartLearner(1, 100, cols, 0.001);
            var timer = new Stopwatch();
            timer.Start();
            var model = sut.Learn(observations, targets);
            timer.Stop();

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Trace.WriteLine("Error: " + error);
            Trace.WriteLine("Time: " + timer.ElapsedMilliseconds);
        }
    }
}
