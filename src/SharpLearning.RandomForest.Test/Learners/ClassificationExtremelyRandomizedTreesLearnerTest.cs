using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Test.Properties;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.RandomForest.Test.Learners
{
    [TestClass]
    public class ClassificationExtremelyRandomizedTreesLearnerTest
    {
        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude_Trees_1()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(1);
            Assert.AreEqual(0.19230769230769232, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude_Trees_5()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(5);
            Assert.AreEqual(0.15384615384615386, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude_Trees_100()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(100);
            Assert.AreEqual(0.076923076923076927, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude_Trees_100_SubSample()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(100, 0.5);
            Assert.AreEqual(0.15384615384615386, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_1()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(1);
            Assert.AreEqual(0.28971962616822428, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_5()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(5);
            Assert.AreEqual(0.070093457943925228, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_100()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(100);
            Assert.AreEqual(0.046728971962616821, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_100_SubSample()
        {
            var error = ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(100, 0.5);
            Assert.AreEqual(0.102803738317757, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationExtremelyRandomizedTreesLearner_Learn_Glass_100_Indices()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationExtremelyRandomizedTreesLearner(100, 1, 100, 1, 0.0001, 1.0,  42, false);
            
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.14485981308411214, error, 0.0000001);
        }

        double ClassificationExtremelyRandomizedTreesLearner_Learn_Glass(int trees, double subSampleRatio = 1.0)
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationExtremelyRandomizedTreesLearner(trees, 1, 100, 1, 0.0001, subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }

        double ClassificationExtremelyRandomizedTreesLearner_Learn_Aptitude(int trees, double subSampleRatio = 1.0)
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var sut = new ClassificationExtremelyRandomizedTreesLearner(trees, 1, 100, 1, 0.0001, subSampleRatio, 42, false);
            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);
            return error;
        }
    }
}
