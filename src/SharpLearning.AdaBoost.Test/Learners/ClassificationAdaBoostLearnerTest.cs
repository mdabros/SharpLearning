using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.AdaBoost.Test.Properties;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using SharpLearning.Containers.Extensions;
using System;
using System.IO;
using System.Linq;
using System.Diagnostics;

namespace SharpLearning.AdaBoost.Test.Learners
{
    [TestClass]
    public class ClassificationAdaBoostLearnerTest
    {
        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_AptitudeData()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var sut = new ClassificationAdaBoostLearner(10);
            
            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_Glass()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationAdaBoostLearner(10, 1, 5);

            var model = sut.Learn(observations, targets);
            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_AptitudeData_SequenceContainNoItemIssue_Solved()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var indices = new int[] { 22, 6, 23, 12 };

            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var sut = new ClassificationAdaBoostLearner(10);
            
            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_Glass_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            var sut = new ClassificationAdaBoostLearner(10, 1, 5);

            var indices = Enumerable.Range(0, targets.Length).ToArray();
            indices.Shuffle(new Random(42));
            indices = indices.Take((int)(targets.Length * 0.7))
                .ToArray();

            var model = sut.Learn(observations, targets, indices);
            var predictions = model.Predict(observations);
            var indexedPredictions = predictions.GetIndices(indices);
            var indexedTargets = targets.GetIndices(indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(indexedTargets, indexedPredictions);

            Assert.AreEqual(0.0, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostLearner_Learn_Glass_Weighted()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();

            // using sample weights to balance the learning based on the frequency of each class in the targets. 
            var classSizes = targets.GroupBy(v => v).ToDictionary(v => v.Key, v => v.Count());
            var weights = targets.Select(v => (double)targets.Length / (double)classSizes[v]).ToArray(); 

            var sut = new ClassificationAdaBoostLearner(10, 1, 2);

            var model = sut.Learn(observations, targets, weights);
            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.ErrorString(targets, predictions);

            // use classification matrix string to ensure all class scores are equal.
            var expected = ";1;2;3;5;6;7;1;2;3;5;6;7\r\n1;40.000;10.000;20.000;0.000;0.000;0.000;57.143;14.286;28.571;0.000;0.000;0.000\r\n2;18.000;49.000;6.000;2.000;1.000;0.000;23.684;64.474;7.895;2.632;1.316;0.000\r\n3;2.000;0.000;15.000;0.000;0.000;0.000;11.765;0.000;88.235;0.000;0.000;0.000\r\n5;0.000;0.000;0.000;13.000;0.000;0.000;0.000;0.000;0.000;100.000;0.000;0.000\r\n6;0.000;4.000;0.000;0.000;5.000;0.000;0.000;44.444;0.000;0.000;55.556;0.000\r\n7;0.000;3.000;0.000;1.000;1.000;24.000;0.000;10.345;0.000;3.448;3.448;82.759\r\nError: 31.776\r\n";

            Assert.AreEqual(expected, actual);
        }
    }
}
