using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.RandomForest.Test.Properties;
using SharpLearning.RandomForest.Learners;
using SharpLearning.Metrics.Classification;
using SharpLearning.Containers;
using System.Linq;
using System.Diagnostics;

namespace SharpLearning.RandomForest.Test.Models
{
    [TestClass]
    public class ClassificationRandomForestModelTest
    {
        [TestMethod]
        public void ClassificationRandomForestModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.23076923076923078, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_Predict_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);
            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };

            var predictions = sut.Predict(observations, indices);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var indexedTargets = targets.GetIndices(indices);
            var error = evaluator.Error(indexedTargets, predictions);

            Assert.AreEqual(0.2, error, 0.0000001);
        }


        [TestMethod]
        public void ClassificationRandomForestModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.GetRow(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.11538461538461539, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.662083945491325 }, { 1, 0.337916054508675 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.613937248577987 }, { 1, 0.386062751422012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.705399166776748 }, { 1, 0.294600833223252 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.711620058501122 }, { 1, 0.288379941498877 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.613937248577987 }, { 1, 0.386062751422012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.762833510211091 }, { 1, 0.237166489788909 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.514910804305026 }, { 1, 0.485089195694974 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.623009667387249 }, { 1, 0.376990332612752 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.779635838730525 }, { 1, 0.220364161269475 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.316381687756688 }, { 1, 0.683618312243312 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.435689560318908 }, { 1, 0.564310439681092 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.705399166776748 }, { 1, 0.294600833223252 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.66195938506723 }, { 1, 0.338040614932771 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.727697821751318 }, { 1, 0.272302178248682 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.355563018809536 }, { 1, 0.644436981190464 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.789714920467501 }, { 1, 0.210285079532498 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.758666843544425 }, { 1, 0.241333156455575 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.560541756685979 }, { 1, 0.439458243314021 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.330798354423354 }, { 1, 0.669201645576645 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.470353720719911 }, { 1, 0.529646279280089 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.682581458212522 }, { 1, 0.317418541787478 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.316381687756688 }, { 1, 0.683618312243312 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.499422305188044 }, { 1, 0.500577694811956 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.789714920467501 }, { 1, 0.210285079532498 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.762833510211091 }, { 1, 0.237166489788909 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.657886004543384 }, { 1, 0.342113995456615 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.11538461538461539, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.662083945491325 }, { 1, 0.337916054508675 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.613937248577987 }, { 1, 0.386062751422012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.705399166776748 }, { 1, 0.294600833223252 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.711620058501122 }, { 1, 0.288379941498877 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.613937248577987 }, { 1, 0.386062751422012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.762833510211091 }, { 1, 0.237166489788909 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.514910804305026 }, { 1, 0.485089195694974 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.623009667387249 }, { 1, 0.376990332612752 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.779635838730525 }, { 1, 0.220364161269475 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.316381687756688 }, { 1, 0.683618312243312 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.435689560318908 }, { 1, 0.564310439681092 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.705399166776748 }, { 1, 0.294600833223252 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.66195938506723 }, { 1, 0.338040614932771 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.727697821751318 }, { 1, 0.272302178248682 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.355563018809536 }, { 1, 0.644436981190464 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.789714920467501 }, { 1, 0.210285079532498 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.758666843544425 }, { 1, 0.241333156455575 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.560541756685979 }, { 1, 0.439458243314021 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.330798354423354 }, { 1, 0.669201645576645 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.470353720719911 }, { 1, 0.529646279280089 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.682581458212522 }, { 1, 0.317418541787478 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.316381687756688 }, { 1, 0.683618312243312 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.499422305188044 }, { 1, 0.500577694811956 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.789714920467501 }, { 1, 0.210285079532498 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.762833510211091 }, { 1, 0.237166489788909 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.657886004543384 }, { 1, 0.342113995456615 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_PredictProbability_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationRandomForestLearner(100, 1, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
            var actual = sut.PredictProbability(observations, indices);

            var indexedTargets = targets.GetIndices(indices);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(indexedTargets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.3, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.662083945491325 }, { 1, 0.337916054508675 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.711620058501122 }, { 1, 0.288379941498877 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.613937248577987 }, { 1, 0.386062751422012 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.762833510211091 }, { 1, 0.237166489788909 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.514910804305026 }, { 1, 0.485089195694974 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.623009667387249 }, { 1, 0.376990332612752 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.779635838730525 }, { 1, 0.220364161269475 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.316381687756688 }, { 1, 0.683618312243312 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.682581458212522 }, { 1, 0.317418541787478 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.316381687756688 }, { 1, 0.683618312243312 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationRandomForestModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 33.4226363879905 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationRandomForestModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationRandomForestLearner(100, 5, 100, 1, 0.0001, 42, 1);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 3.0225981263847426, 9.0435658375257084 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        void Write(ProbabilityPrediction[] predictions)
        {
            var value = "new ProbabilityPrediction[] {";
            foreach (var item in predictions)
            {
                value += "new ProbabilityPrediction(" + item.Prediction + ", new Dictionary<double, double> {";
                foreach (var prob in item.Probabilities)
                {
                    value += "{" + prob.Key + ", " + prob.Value + "}, ";
                }
                value += "}),";
            }
            value += "};";

            Trace.WriteLine(value);
        }
    }
}
