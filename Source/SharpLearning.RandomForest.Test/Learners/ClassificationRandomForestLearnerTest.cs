using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Test.Properties;
using System;
using System.Diagnostics;
using System.IO;

namespace SharpLearning.RandomForest.Test.Learners
{
    [TestClass]
    public class ClassificationRandomForestLearnerTest
    {
        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Aptitude_Trees_1()
        {
            var error = ClassificationRandomLearner_Learn_Aptitude(1);
            Assert.AreEqual(0.34615384615384615, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Aptitude_Trees_5()
        {
            var error = ClassificationRandomLearner_Learn_Aptitude(5);
            Assert.AreEqual(0.30769230769230771, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Aptitude_Trees_100()
        {
            var error = ClassificationRandomLearner_Learn_Aptitude(100);
            Assert.AreEqual(0.26923076923076922, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_1()
        {
            var error = ClassificationRandomForestLearner_Learn_Glass(1);
            Assert.AreEqual(0.45794392523364486, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_5()
        {
            var error = ClassificationRandomForestLearner_Learn_Glass(5);
            Assert.AreEqual(0.5, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationRandomForestLearner_Learn_Glass_100()
        {
            var error = ClassificationRandomForestLearner_Learn_Glass(100);
            Assert.AreEqual(0.22897196261682243, error, 0.0000001);
        }

        double ClassificationRandomForestLearner_Learn_Glass(int trees)
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationRandomForestLearner(trees, 5, 100, 1, 0.0001, 42, 1);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double ClassificationRandomLearner_Learn_Aptitude(int trees)
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationRandomForestLearner(trees, 5, 100, 1, 0.0001, 42);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        [Ignore]
        [TestMethod]
        public void ClassificationRandomLearner_Learn_Timing()
        {
            var rows = 4000;
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

            var sut = new ClassificationRandomForestLearner(20, 5, 100, (int)Math.Sqrt(cols), 0.0001, 42);
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
