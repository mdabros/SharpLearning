using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Containers;
using System.Collections.Generic;
using SharpLearning.Metrics.Regression;
using System.Diagnostics;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.DecisionTrees.Test.Properties;

namespace SharpLearning.DecisionTrees.Test.Learners
{
    [TestClass]
    public class RegressionCartLearnerTest
    {
        [TestMethod]
        public void RegressionCartLearner_Learn()
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

        [Ignore]
        [TestMethod]
        public void RegressionCartLearner_Learn_Timing()
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

            var sut = new RegressionCartLearner(1, 100, 0.001);
            var timer = new Stopwatch();
            timer.Start();
            var model = sut.Learn(observations, targets);
            timer.Stop();

            var predictions = model.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Trace.WriteLine("Error: " + error);
            Trace.WriteLine("Time: " + timer.ElapsedMilliseconds);
        }

    }
}
